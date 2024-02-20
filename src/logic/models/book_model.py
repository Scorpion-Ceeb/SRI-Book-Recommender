class BookModel:

    def __init__(self, title, description, publish_year, author):
        self.tokens = None
        self.title = title
        self.description = description
        self.publish_year = publish_year
        self.author = author

    def add_tokens(self, tokens):
        self.tokens = tokens

