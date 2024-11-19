    result_text = [
        {
            "title": documents[index]["title"],
            "author": documents[index]["author"],
            "link": documents[index]["link"],
            "filename": documents[index]["link"].split('/')[-1],
            "text": documents[index]["text"][:200],  # Show a snippet of text
            "score": score
        }
        for index, score in result
    ]