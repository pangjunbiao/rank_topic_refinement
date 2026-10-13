"""Strict English CSV preparation with fitting-only vocabulary and occurrence masks."""
from __future__ import annotations
import csv
import html
import io
import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from rank_refinement.runtime.artifacts import digest, file_hash, write_json, write_csv
from rank_refinement.data.duplicate_groups import group_records
from rank_refinement.data.vocabulary import fit_features
from rank_refinement.data.completion_masks import make_completion

def clean(text):
    text = html.unescape(unicodedata.normalize('NFKC',text)).replace('’',"'")
    # Anonymization markers are metadata, not passenger topics.
    text = re.sub(r'@?\[(?:agency|location|mayor|police chief|bus|line|train|route|station)[^\]]*\]', ' ',text,flags=re.I)
    text = re.sub(r'https?://\S+|www\.\S+|@[A-Za-z0-9_]+',' ',text)
    text = re.sub(r"\bcan't\b",'cannot',text,flags=re.I)
    text = re.sub(r"\bwon't\b",'will not',text,flags=re.I)
    text = re.sub(r"n't\b",' not',text,flags=re.I)
    text = re.sub(r"\b(\w+)'(?:s|re|ve|ll|d|m)\b",r'\1',text,flags=re.I)
    return ' '.join(text.split()).strip()

def tokenize(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    preserve = {'no','not','nor','never','cannot','must','should','need','needs','want','without'}
    stop = ENGLISH_STOP_WORDS-preserve
    return [w for w in re.findall(r'[a-z]+',text.lower()) if len(w)>=2 and w not in stop]

def read_records(path):
    content = path.read_bytes().decode('utf-8-sig',errors='strict')
    reader = csv.DictReader(io.StringIO(content,newline=''),strict=True)
    required = {'Created_at','Year','Month','Text','latitude','longitude'}
    if reader.fieldnames is None or set(reader.fieldnames)!=required or len(reader.fieldnames)!=6:
        raise ValueError('Unexpected external CSV columns; expected the inspected author schema.')
    rows,empty = [],[]
    for line,r in enumerate(reader,2):
        if None in r or any(v is None for v in r.values()):
            raise ValueError('Malformed external CSV record '+str(line))
        if not r['Text'].strip():
            empty.append(line);continue
        text = clean(r['Text'])
        rows.append({'record_id':f'external_transit:{line}', 'source':'external_transit',
            'source_file':path.name,'source_id':str(line),'normalized_body':text,
            'comparison_text':' '.join(re.findall(r'[a-z0-9]+',text.lower())),
            'clean_text':text,'timestamp':r['Created_at'],'year':r['Year'],
            'latitude':r['latitude'],'longitude':r['longitude']})
    if not rows:raise ValueError('External CSV has no nonempty text records.')
    return rows,{'raw_rows':len(rows)+len(empty),'nonempty_text_rows':len(rows),
        'empty_text_csv_records':empty,'encoding':'utf-8-sig','multiline_records':sum('\n' in r['Text'] for r in csv.DictReader(io.StringIO(content))),
        'year_counts':dict(sorted(Counter(r['year'] for r in rows).items())),
        'timestamp_min':min(r['timestamp'] for r in rows),'timestamp_max':max(r['timestamp'] for r in rows),
        'input_sha256':file_hash(path),'only_text_used_for_models':True,'coordinate_labels_used':False}

def prepare(path,policy,destination,log=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    records,audit = read_records(path)
    groups,edges = group_records(records,policy['near_duplicates'])
    documents={g['group_id']:{**g['representative'],'group_id':g['group_id'],
                  'members':g['members'],'base_tokens':tokenize(g['representative']['clean_text'])}for g in groups}
    # Stable independent groups; no row of a duplicate group crosses a split.
    ordered=sorted(documents);rng=np.random.default_rng(policy['split_seed']);rng.shuffle(ordered)
    n=len(ordered);nfit=int(n*.7);ndev=int(n*.1)
    ids={'fit':sorted(ordered[:nfit]),'development':sorted(ordered[nfit:nfit+ndev]),'test':sorted(ordered[nfit+ndev:])}
    for split,selected in ids.items():
        for g in selected:documents[g]['split']=split
    tokens,metadata=fit_features(documents,ids['fit'],policy['phrases'],policy['vocabulary'],destination)
    for g in documents:documents[g]['tokens']=tokens[g]
    masks={g:make_completion(g,tokens[g],metadata['word_to_id'],policy['completion'])for g in sorted(documents)}
    # English phrases must be displayed as separate words to the English encoder.
    for g in masks:
        masks[g]['observed_text']=' '.join(tokens[g][i].replace('_',' ')for i in masks[g]['observed_positions'])
    v=len(metadata['tokens'])
    def counts(selected,observed=False):
        array=np.zeros((len(selected),v),dtype=np.float32)
        for row,g in enumerate(selected):
            words=masks[g]['observed_ids']if observed else [metadata['word_to_id'][w]for w in tokens[g]if w in metadata['word_to_id']]
            for w in words:array[row,w]+=1
        return array
    model_fit=[g for g in ids['fit']if any(w in metadata['word_to_id']for w in tokens[g])]
    fit_observed=[g for g in model_fit if masks[g]['eligible']]
    development=[g for g in ids['development']if masks[g]['eligible']]
    test=[g for g in ids['test']if masks[g]['eligible']]
    def selected_records(selected):return [documents[g]for g in selected]
    coverage=[]
    for split,selected in ids.items():
        total=sum(len(tokens[g])for g in selected);retained=sum(sum(w in metadata['word_to_id']for w in tokens[g])for g in selected)
        coverage.append({'split':split,'independent_groups':len(selected),'lexical_tokens':total,
            'in_vocabulary_tokens':retained,'in_vocabulary_fraction':retained/total if total else 0.,
            'completion_eligible_groups':sum(masks[g]['eligible']for g in selected),
            'completion_eligible_fraction':sum(masks[g]['eligible']for g in selected)/len(selected)if selected else 0.})
    audit.update(independent_groups=n,duplicate_edges=len(edges),vocabulary=v,
        eligible_words_df20=sum(x>=20 for x in metadata['document_frequency']),
        split_group_counts={s:len(gs)for s,gs in ids.items()},coverage=coverage,
        near_duplicate_grouping_before_split=True,vocabulary_and_phrases_fit_on='external_fit_only',
        empty_fitting_bows_excluded=len(ids['fit'])-len(model_fit),
        scope='external corpus suitability; no model results')
    write_json(destination/'data_audit.json',audit);write_json(destination/'split_group_ids.json',ids)
    write_json(destination/'duplicate_groups.json',[{k:g[k]for k in ['group_id','members']}for g in groups])
    write_json(destination/'duplicate_edges.json',edges);write_json(destination/'feature_records.json',selected_records(sorted(documents)))
    write_json(destination/'completion_masks.json',masks)
    write_csv(destination/'coverage.csv',coverage,list(coverage[0]))
    from rank_refinement.runtime.artifacts import atomic_bytes
    buffer=io.BytesIO();np.savez_compressed(buffer,fit=counts(model_fit),fit_observed=counts(fit_observed,True),
        development_observed=counts(development,True),test_observed=counts(test,True))
    atomic_bytes(destination/'counts.npz',buffer.getvalue())
    if log:
        log.info('EXTERNAL DATA | text rows=%d | independent groups=%d | V=%d | df20=%d',len(records),n,v,audit['eligible_words_df20'])
        for r in coverage:log.info('COVERAGE | %s | groups=%d | IV=%.3f | completion=%d/%d',r['split'],r['independent_groups'],r['in_vocabulary_fraction'],r['completion_eligible_groups'],r['independent_groups'])
    errors=[]
    if v<100:errors.append('fewer than 100 fitting-supported vocabulary words')
    if len(model_fit)<60:errors.append('fewer than 60 nonempty fitting groups')
    if len(fit_observed)<policy['prediction']['context_count']:errors.append('fewer than 64 eligible fitting contexts')
    if len(development)<20:errors.append('fewer than 20 development completion groups')
    if len(test)<50:errors.append('fewer than 50 test completion groups')
    audit['engineering_data_gate_passed']=not errors;audit['data_gate_failures']=errors
    write_json(destination/'data_audit.json',audit)
    return {'fit_ids':model_fit,'fit_observed_ids':fit_observed,'dev_ids':development,'test_ids':test,
        'fit':counts(model_fit),'fit_observed':counts(fit_observed,True),'dev_observed':counts(development,True),
        'test_observed':counts(test,True),'fit_records':selected_records(model_fit),
        'fit_coherence_records':selected_records(ids['fit']),
        'fit_observed_records':selected_records(fit_observed),'dev_records':selected_records(development),
        'test_records':selected_records(test),
        'dev_coherence_records':selected_records(ids['development']),'test_coherence_records':selected_records(ids['test']),
        'masks':masks,'metadata':metadata,'vocabulary':metadata['tokens'],'audit':audit,'all_split_ids':ids}
