# autocomplete-jobnames

The home page to job-name search app.

![homepage](https://github.com/Vishaldawar/autocomplete-jobnames/blob/main/pictures/homepage.png)

The App allows users to just see similar job suggestions as-in how google search will show you results and also when clicked on the "details" tab, it will show a structured table having more information about the jobs matched with the keyword user searched for.

![suggestions](https://github.com/Vishaldawar/autocomplete-jobnames/blob/main/pictures/suggestions.png)

The App also gets powered by three different searching algorithms
- A basic string matching based on Levenshtein distance
- Word2Vec model trained on job names extracted and then matched on cosine similarity
- LLM based tokenization and then matched on cosine similarity

![string_match](https://github.com/Vishaldawar/autocomplete-jobnames/blob/main/pictures/string_match.png)

![embedding_match](https://github.com/Vishaldawar/autocomplete-jobnames/blob/main/pictures/embedding_match.png)

![llm_similarity_match](https://github.com/Vishaldawar/autocomplete-jobnames/blob/main/pictures/llm_similarity_match.png)