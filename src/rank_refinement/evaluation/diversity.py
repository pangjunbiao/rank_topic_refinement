def topic_diversity(topics, expected_words=10):
    if not topics or any(len(t) != expected_words or len(set(t)) != expected_words for t in topics):
        raise ValueError("Every topic needs the declared number of genuine distinct words.")
    return len({word for topic in topics for word in topic}) / (len(topics) * expected_words)
