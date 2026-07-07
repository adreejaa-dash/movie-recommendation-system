"""
generate_sample_data.py
Creates minimal TMDB-compatible CSVs in data/ for testing without Kaggle download.
Run: python generate_sample_data.py
"""

import os, json
import pandas as pd

os.makedirs("data", exist_ok=True)

MOVIES = [
    (299536, "Avengers: Infinity War", "Action|Adventure|Science Fiction", "thanos|infinity gauntlet|superhero", 8.3, 12000, "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos."),
    (299534, "Avengers: Endgame", "Action|Adventure|Science Fiction", "time travel|superhero|avengers", 8.3, 15000, "After the devastating events of Infinity War, the Avengers assemble once more to undo Thanos's actions."),
    (155,    "The Dark Knight", "Action|Crime|Drama|Thriller", "joker|batman|gotham|vigilante", 8.5, 18000, "Batman raises the stakes in his war on crime with the help of Lt. Gordon and DA Harvey Dent."),
    (272,    "Batman Begins", "Action|Crime|Drama", "batman|gotham|fear|origin", 7.7, 10000, "A young Bruce Wayne fears his own rage, retreating to criminal underworld to better understand it."),
    (49026,  "The Dark Knight Rises", "Action|Crime|Drama|Thriller", "bane|batman|gotham|mask", 8.0, 13000, "Eight years after the Joker's reign of anarchy, the Dark Knight must return when Bane forces Gotham into chaos."),
    (27205,  "Inception", "Action|Science Fiction|Thriller", "dream|heist|subconscious|mind", 8.4, 14000, "A thief who steals corporate secrets through the use of dream-sharing technology is given the task of planting an idea."),
    (157336, "Interstellar", "Adventure|Drama|Science Fiction", "space|black hole|gravity|time", 8.3, 11000, "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."),
    (550,    "Fight Club", "Drama|Thriller", "anarchism|fighting|soap|rules", 8.4, 13000, "An insomniac office worker forms an underground fight club with a nihilistic soap salesman."),
    (13,     "Forrest Gump", "Comedy|Drama|Romance", "disability|vietnam|chocolate", 8.5, 15000, "The history of the United States from the 1950s to the 70s unfolds from the perspective of an Alabama man."),
    (238,    "The Godfather", "Crime|Drama", "mafia|family|don|mob", 8.7, 14000, "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son."),
    (240,    "The Godfather Part II", "Crime|Drama", "mafia|family|sicily|young", 9.0, 10000, "The early life and career of Vito Corleone is portrayed while his son Michael expands the family business."),
    (278,    "The Shawshank Redemption", "Crime|Drama", "prison|freedom|hope|banker", 9.3, 16000, "Two imprisoned men bond over years, finding solace and eventual redemption through acts of common decency."),
    (680,    "Pulp Fiction", "Crime|Drama|Thriller", "hitmen|nonlinear|boxer|briefcase", 8.5, 13000, "The lives of two mob hitmen, a boxer, a gangster, and his wife intertwine in four tales of violence and redemption."),
    (424,    "Schindler's List", "Drama|History|War", "holocaust|jews|rescue|factory", 8.9, 12000, "A businessman risks everything to save Jewish refugees during the Holocaust."),
    (769,    "GoodFellas", "Crime|Drama", "mafia|mob|money|gangster", 8.5, 10000, "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen."),
    (11,     "Star Wars", "Action|Adventure|Science Fiction", "space|force|jedi|rebellion", 8.2, 11000, "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee, and two droids."),
    (1891,   "The Empire Strikes Back", "Action|Adventure|Science Fiction", "darth vader|force|jedi|rebel", 8.4, 10000, "After the rebels are overpowered by the Empire, Luke Skywalker begins his Jedi training with Yoda."),
    (78,     "Blade Runner", "Science Fiction|Thriller|Drama", "android|dystopia|replicant|future", 8.1, 9000, "A blade runner must pursue and terminate four replicants who have escaped to Earth."),
    (335984, "Blade Runner 2049", "Science Fiction|Drama|Thriller", "android|replicant|memory|future", 8.0, 8000, "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard."),
    (603,    "The Matrix", "Action|Science Fiction", "simulation|hacker|virtual reality|rebels", 8.7, 14000, "A computer hacker learns from mysterious rebels about the true nature of his reality."),
    (604,    "The Matrix Reloaded", "Action|Science Fiction|Thriller", "simulation|rebels|oracle|choice", 7.2, 10000, "Neo and the rebel leaders estimate they have 72 hours until 250,000 probes discover Zion."),
    (605,    "The Matrix Revolutions", "Action|Science Fiction|Thriller", "machine|choice|war|simulation", 6.7, 9000, "The human city of Zion defends itself against the massive invasion of the machines."),
    (101,    "Moulin Rouge!", "Drama|Music|Romance", "musical|love|paris|cabaret", 7.6, 7000, "A poor Bohemian poet falls in love with a beautiful cabaret dancer at the Moulin Rouge."),
    (120,    "The Lord of the Rings: The Fellowship of the Ring", "Adventure|Fantasy|Action", "ring|hobbit|wizard|dark lord", 8.8, 15000, "A meek Hobbit sets out on a fantastic journey along with eight companions to destroy a powerful ring."),
    (121,    "The Lord of the Rings: The Two Towers", "Adventure|Fantasy|Action", "ents|gollum|rohan|war", 8.7, 14000, "While Frodo and Samwise edge closer to Mordor, the divided Fellowship makes a stand against Sauron's new ally."),
    (122,    "The Lord of the Rings: The Return of the King", "Adventure|Fantasy|Action", "mordor|hobbits|aragorn|crowning", 8.9, 15000, "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze as Frodo and Sam approach Mount Doom."),
    (8960,   "Hancock", "Action|Comedy|Crime|Drama|Fantasy", "superhero|antihero|redemption|power", 6.4, 7000, "A superhero with a bad attitude is persuaded by a publicist to change his image."),
    (1726,   "Iron Man", "Action|Science Fiction|Adventure", "superhero|suit|billionaire|arc reactor", 7.9, 12000, "A billionaire industrialist and genius inventor who is kidnapped and builds a mechanical suit to escape."),
    (10138,  "Iron Man 2", "Action|Adventure|Science Fiction", "superhero|armor|palladium|competition", 7.0, 10000, "Tony Stark must contend with deadly issues involving the Iron Man technology."),
    (68721,  "Iron Man 3", "Action|Adventure|Science Fiction", "mandarin|extremis|armor|suits", 7.2, 10000, "Tony Stark is left to survive by his own devices when Mandarin attacks and destroys his mansion."),
]

CAST_MAP = {
    "Avengers: Infinity War": ["Robert Downey Jr.", "Chris Evans", "Josh Brolin", "Scarlett Johansson", "Chris Hemsworth"],
    "Avengers: Endgame": ["Robert Downey Jr.", "Chris Evans", "Mark Ruffalo", "Chris Hemsworth", "Scarlett Johansson"],
    "The Dark Knight": ["Christian Bale", "Heath Ledger", "Aaron Eckhart", "Michael Caine", "Gary Oldman"],
    "Batman Begins": ["Christian Bale", "Michael Caine", "Gary Oldman", "Katie Holmes", "Liam Neeson"],
    "The Dark Knight Rises": ["Christian Bale", "Tom Hardy", "Anne Hathaway", "Gary Oldman", "Michael Caine"],
    "Inception": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Ellen Page", "Tom Hardy", "Ken Watanabe"],
    "Interstellar": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Michael Caine", "Matt Damon"],
    "Fight Club": ["Brad Pitt", "Edward Norton", "Helena Bonham Carter", "Meat Loaf", "Jared Leto"],
    "Forrest Gump": ["Tom Hanks", "Robin Wright", "Gary Sinise", "Sally Field", "Mykelti Williamson"],
    "The Godfather": ["Marlon Brando", "Al Pacino", "James Caan", "Richard Castellano", "Robert Duvall"],
    "The Godfather Part II": ["Al Pacino", "Robert Duvall", "Diane Keaton", "Robert De Niro", "Talia Shire"],
    "The Shawshank Redemption": ["Tim Robbins", "Morgan Freeman", "Bob Gunton", "William Sadler", "Clancy Brown"],
    "Pulp Fiction": ["John Travolta", "Samuel L. Jackson", "Uma Thurman", "Harvey Keitel", "Tim Roth"],
    "Schindler's List": ["Liam Neeson", "Ben Kingsley", "Ralph Fiennes", "Caroline Goodall", "Jonathan Sagall"],
    "GoodFellas": ["Robert De Niro", "Ray Liotta", "Joe Pesci", "Lorraine Bracco", "Paul Sorvino"],
    "Star Wars": ["Mark Hamill", "Harrison Ford", "Carrie Fisher", "Alec Guinness", "Peter Cushing"],
    "The Empire Strikes Back": ["Mark Hamill", "Harrison Ford", "Carrie Fisher", "Billy Dee Williams", "Anthony Daniels"],
    "Blade Runner": ["Harrison Ford", "Rutger Hauer", "Sean Young", "Edward James Olmos", "M. Emmet Walsh"],
    "Blade Runner 2049": ["Ryan Gosling", "Harrison Ford", "Ana de Armas", "Jared Leto", "Robin Wright"],
    "The Matrix": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving", "Gloria Foster"],
    "The Matrix Reloaded": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving", "Jada Pinkett Smith"],
    "The Matrix Revolutions": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving", "Mary Alice"],
    "Moulin Rouge!": ["Nicole Kidman", "Ewan McGregor", "John Leguizamo", "Jim Broadbent", "Richard Roxburgh"],
    "The Lord of the Rings: The Fellowship of the Ring": ["Elijah Wood", "Ian McKellen", "Orlando Bloom", "Liv Tyler", "Viggo Mortensen"],
    "The Lord of the Rings: The Two Towers": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen", "Orlando Bloom", "Sean Astin"],
    "The Lord of the Rings: The Return of the King": ["Elijah Wood", "Ian McKellen", "Viggo Mortensen", "Sean Astin", "Orlando Bloom"],
    "Hancock": ["Will Smith", "Charlize Theron", "Jason Bateman", "Eddie Marsan", "David Mattey"],
    "Iron Man": ["Robert Downey Jr.", "Gwyneth Paltrow", "Jeff Bridges", "Terrence Howard", "Shaun Toub"],
    "Iron Man 2": ["Robert Downey Jr.", "Gwyneth Paltrow", "Don Cheadle", "Scarlett Johansson", "Mickey Rourke"],
    "Iron Man 3": ["Robert Downey Jr.", "Gwyneth Paltrow", "Don Cheadle", "Guy Pearce", "Rebecca Hall"],
}

DIRECTOR_MAP = {
    "Avengers: Infinity War": "Anthony Russo",
    "Avengers: Endgame": "Anthony Russo",
    "The Dark Knight": "Christopher Nolan",
    "Batman Begins": "Christopher Nolan",
    "The Dark Knight Rises": "Christopher Nolan",
    "Inception": "Christopher Nolan",
    "Interstellar": "Christopher Nolan",
    "Fight Club": "David Fincher",
    "Forrest Gump": "Robert Zemeckis",
    "The Godfather": "Francis Ford Coppola",
    "The Godfather Part II": "Francis Ford Coppola",
    "The Shawshank Redemption": "Frank Darabont",
    "Pulp Fiction": "Quentin Tarantino",
    "Schindler's List": "Steven Spielberg",
    "GoodFellas": "Martin Scorsese",
    "Star Wars": "George Lucas",
    "The Empire Strikes Back": "Irvin Kershner",
    "Blade Runner": "Ridley Scott",
    "Blade Runner 2049": "Denis Villeneuve",
    "The Matrix": "Lana Wachowski",
    "The Matrix Reloaded": "Lana Wachowski",
    "The Matrix Revolutions": "Lana Wachowski",
    "Moulin Rouge!": "Baz Luhrmann",
    "The Lord of the Rings: The Fellowship of the Ring": "Peter Jackson",
    "The Lord of the Rings: The Two Towers": "Peter Jackson",
    "The Lord of the Rings: The Return of the King": "Peter Jackson",
    "Hancock": "Peter Berg",
    "Iron Man": "Jon Favreau",
    "Iron Man 2": "Jon Favreau",
    "Iron Man 3": "Shane Black",
}


def make_json_list(items, key="name"):
    return json.dumps([{key: i} for i in items])


def make_crew_json(director):
    return json.dumps([{"name": director, "job": "Director"}])


rows_movies = []
rows_credits = []

for movie_id, title, genres_str, keywords_str, vote_avg, vote_cnt, overview in MOVIES:
    genres_list = [g.strip() for g in genres_str.split("|")]
    keywords_list = keywords_str.split("|")
    rows_movies.append({
        "id": movie_id,
        "title": title,
        "genres": make_json_list(genres_list),
        "keywords": make_json_list(keywords_list),
        "overview": overview,
        "vote_average": vote_avg,
        "vote_count": vote_cnt,
        "popularity": vote_cnt / 1000,
    })
    cast_list = CAST_MAP.get(title, [])
    crew_json = make_crew_json(DIRECTOR_MAP.get(title, "Unknown"))
    rows_credits.append({
        "movie_id": movie_id,
        "title": title,
        "cast": make_json_list(cast_list),
        "crew": crew_json,
    })

pd.DataFrame(rows_movies).to_csv("data/tmdb_5000_movies.csv", index=False)
pd.DataFrame(rows_credits).to_csv("data/tmdb_5000_credits.csv", index=False)
print(f"✅ Generated {len(rows_movies)} movies in data/")
