import pdb
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # view overall statistics
    start = time.time()
    train = pd.read_csv('data/train.csv')
    print("Took {}s to open".format(time.time() - start))
    print("DF shape is {}".format(train.shape))
    test = pd.read_csv('data/test.csv')

    start = time.time()
    print(train.describe().T)
    print("Took {}s to open".format(time.time() - start))


    print("\nProportion of positive to negative examples")
    print(train['is_listened'].value_counts())

    print("\nUnique value count for training set")
    for var in train.columns.tolist():
        print("{} has {} unique values".format(var, train[var].nunique()))

    print("\nUnique value count for test set")
    for var in test.columns.tolist():
        print("{} has {} unique values".format(var, test[var].nunique()))


    print('\nUniqueness between train and test')
    # check uniqueness of variables
    for var in train.columns.tolist():
        if var != 'is_listened':
            match =  set(test[var].unique()) - set(train[var].unique())
            count = len(match)
            match = False if match else True

            if not match:
                print('{} : {} -> {} don\'t match'.format(var, match,count))
            else:
                print('{} : {}'.format(var, match))
    
    # look at song distribution
    print("\nLooking at media_id distribution")
    song = train['media_id']
    dist_song = song.value_counts().tolist()
    songdict = {}
    for count in dist_song:
        if songdict.get(count):
            songdict[count] += 1
        else:
            songdict[count] = 1

    count = 0
    for i in range(1, 21):
        count += songdict.get(i)
        print('{} have only {} feedback'.format(songdict.get(i), i))
    
    print('{}/{} songs have under 20 feedback'.format(count, len(dist_song)))
    
    # look at genre distribution
    print("\nLooking at genre_id distribution")
    genre = train['genre_id']
    dist_genre = genre.value_counts().tolist()
    gdict = {}
    for count in dist_genre:
        if gdict.get(count):
            gdict[count] += 1
        else:
            gdict[count] = 1

    count = 0
    for i in range(1, 21):
        count += gdict.get(i)
        print('{} have only {} feedback'.format(gdict.get(i), i))
    
    print('{}/{} genre have under 20 feedback'.format(count, len(dist_genre)))

    # look at album distribution
    print("\nLooking at album_id distribution")
    album = train['album_id']
    dist_album = album.value_counts().tolist()
    adict = {}
    for count in dist_album:
        if adict.get(count):
            adict[count] += 1
        else:
            adict[count] = 1

    count = 0
    for i in range(1, 21):
        count += adict.get(i)
        print('{} have only {} feedback'.format(adict.get(i), i))
    
    print('{}/{} album have under 20 feedback'.format(count, len(dist_album)))

    # look at artist distribution
    print("\nLooking at artist_id distribution")
    artist = train['artist_id']
    dist_artist = artist.value_counts().tolist()
    artist_dict = {}
    for count in dist_artist:
        if artist_dict.get(count):
            artist_dict[count] += 1
        else:
            artist_dict[count] = 1

    count = 0
    for i in range(1, 21):
        count += artist_dict.get(i)
        print('{} have only {} feedback'.format(artist_dict.get(i), i))
    
    print('{}/{} artist have under 20 feedback'.format(count, len(dist_artist)))



    pdb.set_trace()
