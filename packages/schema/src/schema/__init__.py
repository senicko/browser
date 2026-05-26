class Document:
    """
    Document represents a single document from an article.
    """

    title: str
    text: str
    url: str

    def __init__(self, title: str, text: str):
        self.title = title
        self.text = text
        self.url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
