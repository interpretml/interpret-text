import requests
import os
import pickle
import random
from tqdm import tqdm
from nltk.corpus import words


def sample_songs():
    endpoint_url = "https://query.wikidata.org/sparql"

    query = f"""
    SELECT DISTINCT ?song ?songLabel ?artist ?artistLabel (SAMPLE(?popularity) AS ?popularity_count)
    WHERE {{
      ?song wdt:P31 wd:Q7366; # Instance of song
            wdt:P175 ?artist; # Performer
            wikibase:sitelinks ?popularity.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?song ?songLabel ?artist ?artistLabel
    HAVING (SAMPLE(?popularity) >= 5)
    """

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.reason}")

    results = response.json().get("results", {}).get("bindings", [])
    songs_info = [{"song_name": result["songLabel"]["value"], "artist_name": result["artistLabel"]["value"], "popularity": int(result["popularity_count"]["value"])} for result in results]

    return songs_info

def sample_universities():
    endpoint_url = "https://query.wikidata.org/sparql"

    query = f"""
    SELECT DISTINCT ?university ?universityLabel (SAMPLE(?f_year) AS ?founding_year) (SAMPLE(?cityLabel) AS ?city) (SAMPLE(?popularity) AS ?popularity_count)
    WHERE {{
      ?university wdt:P31 wd:Q3918. # Instance of university
      OPTIONAL {{ ?university wdt:P571 ?founding_date. BIND(YEAR(?founding_date) AS ?f_year) }}
      OPTIONAL {{ ?university wdt:P131 ?city. }}
      OPTIONAL {{ ?university wikibase:sitelinks ?popularity. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?university ?universityLabel
    HAVING (SAMPLE(?popularity) >= 5)
    """

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.reason}")

    results = response.json().get("results", {}).get("bindings", [])
    universities_info = []

    for result in results:
        university_info = {
            "university_name": result.get("universityLabel", {}).get("value", ""),
            "founding_year": int(result.get("founding_year", {}).get("value", "0")) if result.get("founding_year") else None,
            "city": result.get("city", {}).get("value", "") if result.get("city") else None,
            "popularity": int(result.get("popularity_count", {}).get("value", "0")) if result.get("popularity_count") else None
        }
        universities_info.append(university_info)

    return universities_info


def sample_basketball_players():
    endpoint_url = "https://query.wikidata.org/sparql"

    query = """
    SELECT DISTINCT ?player ?playerLabel ?birth_year ?birthplaceLabel ?popularity WHERE {
      ?player wdt:P31 wd:Q5;                    # <- instance of human
             wdt:P641 wd:Q5372;                # <- sport: basketball
             wdt:P106 wd:Q3665646;             # <- occupation: basketball player
             wikibase:sitelinks ?popularity.   # <- popularity (site links)
      FILTER(?popularity >= 5)                 # <- filter by popularity > 20
      OPTIONAL { ?player wdt:P569 ?birth_date. BIND(YEAR(?birth_date) AS ?birth_year) } # <- birth year
      OPTIONAL { ?player wdt:P19 ?birthplace. } # <- birthplace
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
      }
    }
    """

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.reason}")

    results = response.json().get("results", {}).get("bindings", [])
    players_info = [
        {
            "player_name": result.get("playerLabel", {}).get("value", ""),
            "birth_year": int(result.get("birth_year", {}).get("value", "0")) if result.get("birth_year") else None,
            "birthplace": result.get("birthplaceLabel", {}).get("value", "") if result.get("birthplaceLabel") else None,
            "popularity": int(result.get("popularity", {}).get("value", "0")) if result.get("popularity") else None
        }
        for result in results
    ]

    return players_info


def sample_football_teams():
    endpoint_url = "https://query.wikidata.org/sparql"
    
    query = f"""
    SELECT DISTINCT ?team ?teamLabel (SAMPLE(?f_year) AS ?founding_year) (SAMPLE(?stadiumLabel) AS ?stadium_name) (SAMPLE(?cityLabel) AS ?city) (SAMPLE(?popularity) AS ?popularity_count)
    WHERE {{
      ?team wdt:P31 wd:Q476028. # Instance of football club
      OPTIONAL {{ ?team wdt:P571 ?founding_date. BIND(YEAR(?founding_date) AS ?f_year) }}
      OPTIONAL {{ ?team wdt:P115 ?stadium. }}
      OPTIONAL {{ ?team wdt:P131 ?city. }}
      OPTIONAL {{ ?team wikibase:sitelinks ?popularity. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?team ?teamLabel
    HAVING (SAMPLE(?popularity) >= 5)
    """

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.reason}")

    results = response.json().get("results", {}).get("bindings", [])
    teams_info = []

    for result in results:
        team_info = {
            "team_name": result.get("teamLabel", {}).get("value", ""),
            "founding_year": int(result.get("founding_year", {}).get("value", "0")) if result.get("founding_year") else None,
            "stadium_name": result.get("stadium_name", {}).get("value", "") if result.get("stadium_name") else None,
            "city": result.get("city", {}).get("value", "") if result.get("city") else None,
            "popularity": int(result.get("popularity_count", {}).get("value", "0")) if result.get("popularity_count") else None
        }
        teams_info.append(team_info)

    return teams_info


def load_basketball_players():
    if not os.path.exists("./rare_knowledge/data/basketball_players.pkl"):
        items = sample_basketball_players()
        with open("./rare_knowledge/data/basketball_players.pkl", "wb") as f:
            pickle.dump(items, f)
    
    with open("./rare_knowledge/data/basketball_players.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the basketball player {} was born in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The player was born in"
    for item in items:
        item["name"] = item["player_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["birth_year"]
        item["popularity"] = item["popularity"]
    return items, prompt_fn, prompt_template


def load_football_teams():
    if not os.path.exists("./rare_knowledge/data/football_teams.pkl"):
        items = sample_football_teams()
        with open("./rare_knowledge/data/football_teams.pkl", "wb") as f:
            pickle.dump(items, f)

    with open("./rare_knowledge/data/football_teams.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the football team {} was founded in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The team was founded in"
    for item in items:
        item["name"] = item["team_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["founding_year"]
        item["popularity"] = item["popularity"]
    return items, prompt_fn, prompt_template


def load_fake_football_teams():
    with open("./rare_knowledge/data/fake_football_teams.pkl", "rb") as f:
        items = pickle.load(f)
    prompt_template = "Tell me the year the football team {} was founded in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The team was founded in"
    for item in items:
        item["name"] = item["team_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["founding_year"]
        item["popularity"] = item["popularity"]
    return items, prompt_fn, prompt_template    

def load_songs():
    if not os.path.exists("./rare_knowledge/data/songs.pkl"):
        items = sample_songs()
        with open("./rare_knowledge/data/songs.pkl", "wb") as f:
            pickle.dump(items, f)
        
    with open("./rare_knowledge/data/songs.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the the performer of the song {}"
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The performer is"
    for item in items:
        item["name"] = item["song_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["artist_name"]
        item["popularity"] = item["popularity"]
    return items, prompt_fn, prompt_template


def load_schools():
    if not os.path.exists("./rare_knowledge/data/schools.pkl"):
        items = sample_universities()
        with open("./rare_knowledge/data/schools.pkl", "wb") as f:
            pickle.dump(items, f)
        
    with open("./rare_knowledge/data/schools.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year {} was founded in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: It was founded in"
    for item in items:
        item["name"] = item["university_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["founding_year"]
        item["popularity"] = item["popularity"]
    return items, prompt_fn, prompt_template


def load_counterfact():
    import json
    counterfact = json.load(open("./rare_knowledge/data/counterfact.json"))
    all_items = []
    for item in counterfact:
        prompt = item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"])
        all_items.append({"prompt": prompt, 
                          "label": item["requested_rewrite"]["target_true"]["str"], 
                          "name": item["requested_rewrite"]["subject"],
                          "popularity": -1})

    return all_items, None, None

def conditions_to_prompt(conditions):
    prompt = "Is there a word where:\n"
    constraints = []
    for pos, letter in conditions:
        if pos == 1:
            constraint = f"- The 1st letter is {letter}"
        elif pos == 2:
            constraint = f"- The 2nd letter is {letter}"
        elif pos == 3:
            constraint = f"- The 3rd letter is {letter}"
        else:
            constraint = f"- The {pos}th letter is {letter}"
        prompt += (constraint + "\n")
        constraints.append(constraint)
    return prompt, constraints


def generate_conditions(num_conditions=3, maxpos=10):
    conditions = []
    poses = []
    random_word = random.choice(words.words())
    for _ in range(num_conditions):
        #letter = random.choice(string.ascii_lowercase)
        
        position = random.randint(1, min(maxpos, len(random_word)))
        while position in poses:
            position = random.randint(1, min(maxpos, len(random_word)))
        poses.append(position)
        letter = random_word[position-1].lower()
        conditions.append((position, letter))
    return conditions

def satisfies_conditions(word, conditions):
    for pos, letter in conditions:
        try:
            if word[pos-1] != letter: return False
        except IndexError:
            return False
    return True

def count_satisfying_words(conditions):
    return sum(satisfies_conditions(w, conditions) for w in words.words())

def sample_words(N=10000, num_conds=2):
    items = []
    for k in tqdm(range(N)):
        conditions = generate_conditions(num_conds, maxpos=4)
        prompt, constraints = conditions_to_prompt(conditions)
        popularity = count_satisfying_words(conditions)
        full_prompt = f"User: {prompt}\nAssistant: Yes, the word is"
        items.append({"prompt": full_prompt, "conditions": conditions, "constraints": constraints,
                     "popularity": popularity})    
    return items

def load_word_dataset():
    filename = "./rare_knowledge/data/words_10k.pkl"
    if not os.path.exists(filename):
        items = sample_words()
        with open(filename, "wb") as f:
            pickle.dump(items, f)
    with open(filename, "rb") as f:
        items = pickle.load(f)
    return items, None, None


def load_trivia_qa_dataset():
    filename = "./rare_knowledge/data/trivia_qa.pkl"
    if not os.path.exists(filename):
        raise ValueError()
    with open(filename, "rb") as f:
        items = pickle.load(f)
    return items, None, None


def sample_movies():
    endpoint_url = "https://query.wikidata.org/sparql"

    query = f"""
    SELECT ?movie ?movieLabel ?directorLabel ?popularity WHERE {{
      ?movie wdt:P31 wd:Q11424;                 # <- instance of film
            wdt:P57 ?director;                  # <- director
            wikibase:sitelinks ?popularity.     # <- popularity (site links)
      FILTER(?popularity >= 15)                    # <- filter by popularity > 5
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
      }}
    }}
    """

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.reason}")

    results = response.json().get("results", {}).get("bindings", [])
    movies_info = [
        {
            "movie_name": result.get("movieLabel", {}).get("value", ""),
            "director_name": result.get("directorLabel", {}).get("value", ""),
            "popularity": int(result.get("popularity", {}).get("value", "0")) if result.get("popularity") else None
        }
        for result in results
    ]

    return movies_info


def load_movies():
    filename = "./rare_knowledge/data/movies.pkl"
    if not os.path.exists(filename):
        items = sample_movies()
        with open(filename, "wb") as f:
            pickle.dump(items, f)
    with open(filename, "rb") as f:
        items = pickle.load(f)
    prompt_template = "Tell me the the director of the movie {}."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The director is"
    for item in items:
        item["name"] = item["movie_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["name"]))
        item["label"] = item["director_name"]
    return items, prompt_fn, prompt_template

def load_nobel_city():
    filename = "/home/t-merty/mounts/sandbox-mert/data/nobel_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None

def load_senator_multiconstraint():
    filename = "/home/t-merty/mounts/sandbox-mert/data/senator_multiconstraint_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None

def load_senator_multiconstraint_v2():
    filename = "/home/t-merty/mounts/sandbox-mert/data/senator_multiconstraint_v2_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None

def load_word_startend():
    filename = "/home/t-merty/mounts/sandbox-mert/data/word_startend_multiconstraint_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None

def load_books():
    filename = "/home/t-merty/mounts/sandbox-mert/data/books_multiconstraints.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None


def load_movie_award():
    filename = "/home/t-merty/mounts/sandbox-mert/data/movie_award_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None


def load_senator_assistant():
    filename = "/home/t-merty/mounts/sandbox-mert/data/senator_assistant_data.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None


def load_counterfact():
    filename = "/home/t-merty/mounts/sandbox-mert/data/counterfact.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None   

def load_counterfact_subset(subset):
    filename = f"/home/t-merty/mounts/sandbox-mert/data/{subset}.pkl"
    items = pickle.load(open(filename, "rb"))
    return items, None, None   