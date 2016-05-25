# ldateach, generate/choose documents to teach topics model
# Copyright (C) 2016 Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


DOCSTR = """ Scrape IMDB top movie synopses and prepare them for LDA """

import pandas
import re
import requests
import os
import math
import time


DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
HEADERS = {'Accept-Language': 'en-US,en;q=0.8'}
TIMEOUT = 2


def _get_movie_data(id):
    '''Returns movie data of given IMDB ID in dictionary format using API
    request.
    '''
    omdb_res = requests.post(
        "http://www.omdbapi.com/?i={0}&plot=full&r=json".format(id),
        headers=HEADERS, timeout=TIMEOUT).json()

    req = 'http://www.imdb.com/title/{}/'.format(id)
    res = requests.get(req, headers=HEADERS, timeout=TIMEOUT)
    # If we cannot find the synopsis on the page, use the one from OMDB.
    # XXX: The synopses from OMDB are often the one-sentence blurbs, rather
    # than the long-form synopses.
    try:
        plot = re.findall(r'itemprop="description">\s+<p>\s+(.*)<em', res.text)[0]
    except IndexError:
        plot = omdb_res['Plot']

    assert isinstance(plot, str)

    return {'Title': omdb_res['Title'],
            't': plot,
            'Genre': omdb_res['Genre']}


def _scrape_movie_ids(n_films=250):
    '''Scrapes the IDs of IMDB's Top 250 movies.'''

    if n_films > 1000:
        raise NotImplementedError("Top n > 1000 not supported.")

    idxs = []
    stepsize = 100  # currently shows 100 movies per page
    n_steps = math.ceil(float(n_films)/stepsize)
    start_idx = 1

    for i in range(n_steps):
        end_idx = start_idx + stepsize - 1
        n_get = min(stepsize, stepsize-end_idx+n_films)

        req = "http://www.imdb.com/search/title?groups=top_1000" +\
              "&sort=user_rating&start={}&view=simple".format(start_idx)
        source = requests.get(req, headers=HEADERS)
        res = re.findall(r'.<a href="/title/(\w+)/', source.text)[:n_get]

        start_idx += stepsize

        idxs.extend(res)

    return idxs


def scrape_imdb(n_films=250, filename='imdb-top-250.csv'):
    ids = _scrape_movie_ids(n_films)
    movie_data = []

    print("\nWorking with API ({} titles)...\n".format(len(ids)))

    # Loop through IDs and make an API request for each one.
    for id in ids:
        try:
            response = _get_movie_data(id)
            if all(item in response for item in ['Title', 'Genre', 'Plot']):
                movie_data.append(response)
                title = response['Title']
                print("     * Request '{}' succeeded.".format(title))
            else:
                print("     * Request #{} failed (unexpected output).".format(
                    ids.index(id) + 1))
        except Exception as e:
            print("     * Request #{} failed.".format(ids.index(id) + 1))
            print("         + EXCEPTION: {}".format(e))
            time.sleep(1.5)

    # Write final data to a table.
    df = pandas.DataFrame(movie_data)
    df.to_csv(filename, encoding='utf-8')


if __name__ == "__main__":
    import argparse

    filename = os.path.join(DATA_DIR, 'imdb.csv')

    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-n', '--n_films', type=int, default=250)
    parser.add_argument('-f', '--filename', type=str, default=filename)
    args = parser.parse_args()

    scrape_imdb(**vars(args))
