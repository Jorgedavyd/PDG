# PDG

## Description:

**PDG** (Potential Distribution Generator) is an AI enhanced specie distribution generator suitable for all kind.

## How to use:

If you are trying to run it cloning the repository, make sure your environment has the correct package versions and packages shown in `requirements.txt`:

`pip install -r requirements.txt`


If you are using the executable version, you can follow these steps as well.

1. Create a `.csv` or `.xlsx` with the columns `Longitude, Latitude, Presence` in that order.
   1. `Longitude`: Longitudes of the presence or absence place.
   2. `Latitude`: Latitudes of the presence or absence place
   3. `Presence`: 1 or 0 (1 if presence, 0 if absence)
      Example (more examples on `data` folder):
      | Longitude | Latitude | Presence |
      |----------|----------|----------|
      | -55.8314992232851 | -20.4883331282854 | 1 |
      | -55.2158638241533 | -20.0939445169726 | 1 |
      | -47.8817725780146 | -15.9205169869386 | 0 |
2. Save the file on `./PDG/data` folder.
3. Now you can access other file rather than the example.
4. Execute `main.py`
