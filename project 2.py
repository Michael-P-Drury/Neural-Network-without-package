# import all of the packages that I will be using
# pandas - used to collect the data from the large datasets

import pandas as pd
import numpy as np
import time
import random
import sklearn

def building_the_nn(X, Y, m):

    """

    :param X:
    :param Y:
    :return:

    """



    from sklearn.preprocessing import OneHotEncoder
    # this imports One Hot Encoder that is a package that I will use to turn all of the string data into ones and zeros
    # so that I can pass them into the neural network
    new_df = X[['transmission', 'fuelType']]
    # this selects the columns in the database that will need to be one hot encoded
    x = OneHotEncoder().fit_transform(new_df).toarray()
    ohe = pd.DataFrame(x)
    X = pd.concat([X, ohe], axis=1, join="inner")
    X = X.drop(columns=['transmission', 'fuelType'])
    # this gets rid of the string versions of the columns as they are no longer needed
    X = X.values
    num_of_collums = X.shape[0]
    size_of_train_data = int(num_of_collums // 1.5)
    # this finds out the number of entries of data that will be used for the training set

    Xtrain = X[:size_of_train_data]
    # this splits the data up into a training set and a test set with a split of 2 thirds and a third but I will adjust
    # that when attemptimg to find the optimal split

    # Xtrain = np.reshape(Xtrain, (Xtrain.shape[1], size_of_train_data)) # reshapes the 2d array so rows are number of input values but wrong lol

    Xtrain = np.array(Xtrain).T # reshapes the 2d array so rows are number of input values but right

    number_of_models = len(np.unique(Y))

    Y = Y.values

    Ytrain = Y[:size_of_train_data]

    Ytrain = np.array(Ytrain).T

    learning_rate = 0.0005

    amount_of_training_times = 100000

    car_models = []

    Xtrain[0] = Xtrain[0] / 10000
    Xtrain[1] = Xtrain[1] / 1000
    Xtrain[2] = Xtrain[2] / 10000
    Xtrain[3] = Xtrain[3] / 100


    for i in range(0, Ytrain.shape[1]):
        if Ytrain[0,i] in car_models:
            z = 1
        else:
            car_models.append(Ytrain[0,i])

    number_of_models = len(car_models)

    Ytrain2 = np.zeros((number_of_models, Ytrain.shape[1]))

    for i in range(0, Ytrain.shape[1]):

        x = car_models.index(Ytrain[0, i])

        Ytrain2 [x, i] = 1




    def train(Y, X, W, W1, W2, B, B1, B2, LR):

        z = np.dot(W, X) + B
        A = 1/(1+np.exp(-z))

        z1 = np.dot(W1, A) + B1
        A1 = 1/(1+np.exp(-z1))

        z2 = np.dot(W2, A1) + B2
        A2 = 1/(1+np.exp(-z2))

        da = (np.exp(-z)/(np.exp(-z)+1)**2)
        da1 = (np.exp(-z1)/(np.exp(-z1)+1)**2)

        dz2 = np.subtract(A2, Y)
        dw2 = (1 / X.shape[0]) * np.dot(dz2, A1.T)
        db2 = (1/X.shape[0]) * np.sum(dz2, axis = 1, keepdims = True)

        W2 = W2 - LR*dw2
        B2 = B2 - LR*db2

        dz1 = np.dot(W2.T, dz2) * da1
        dw1 = (1/X.shape[0]) * np.dot(dz1, A.T)
        db1 = (1/X.shape[0]) * np.sum(dz1, axis = 1, keepdims = True)

        W1 = W1 - LR * dw1
        B1 = B1 - LR * db1

        dz = np.dot(W1.T, dz1) * da
        dw = (1/X.shape[0]) * np.dot(dz, X.T)
        db = (1 / X.shape[0]) * np.sum(dz, axis=1, keepdims=True)

        W = W - LR * dw
        B = B - LR * db

        return W, W1, W2, B, B1, B2, A2


    B = np.zeros ((((X.shape[1]*2)//3),1))

    B1 = np.zeros ((((X.shape[1]*2)//3),1))

    B2 = 0

    W = np.random.randn((X.shape[1]*2)//3, X.shape[1]) * 0.0001

    W1 = np.random.randn((X.shape[1] * 2) // 3, (X.shape[1] * 2) // 3) * 0.0001

    W2 = np.random.randn(number_of_models, (X.shape[1] * 2) // 3) * 0.0001


    train(Ytrain2, Xtrain, W, W1, W2, B, B1, B2, learning_rate)

    for i in range (0, amount_of_training_times):
        t_rain = train(Ytrain2, Xtrain, W, W1, W2, B, B1, B2, learning_rate)

        W = t_rain[0]
        W1 = t_rain[1]
        W2 = t_rain[2]
        B = t_rain[3]
        B1 = t_rain[4]
        B2 = t_rain[5]
        guess = t_rain[6]



        model_guess = np.zeros((1, guess.shape[1]))


        for i in range (0, guess.shape[1]):
            case = guess[:, i]
            max = case[0]
            max_position = 0
            for j in range (0, guess.shape[0]):
                if case[j] > max:
                    max = case[j]
                    max_position = j
            model_guess[0, i] = max_position

        print('')
        print(model_guess[:, 0])
        print(model_guess[:, 2])
        print('')



def ford(m):
    """

    :param m:
    :return:
    """
    db = pd.read_csv('ford.csv')

    db = db[db["model"].str.contains("Puma") == False]
    db = db[db["model"].str.contains("Mustang") == False]
    db = db[db["model"].str.contains("KA") == False]
    db = db[db["model"].str.contains("Mondeo") == False]
    db = db[db["model"].str.contains("B-MAX") == False]
    db = db[db["model"].str.contains("S-MAX") == False]
    db = db[db["model"].str.contains("Galaxy") == False]
    db = db[db["model"].str.contains("Edge") == False]
    db = db[db["model"].str.contains("Tourneo Custom") == False]
    db = db[db["model"].str.contains("Tourneo Connect") == False]
    db = db[db["model"].str.contains("Tourneo Connect Connect") == False]
    db = db[db["model"].str.contains("Focus") == False]
    db = db[db["model"].str.contains("C-MAX") == False]
    db = db[db["model"].str.contains("EcoSport") == False]
    db = db[db["model"].str.contains("Ka+") == False]
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]

    building_the_nn(X, Y, m)


def audi(m):
    db = pd.read_csv('audi.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def bmw(m):
    db = pd.read_csv('bmw.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def hyundi(m):
    db = pd.read_csv('hyundi.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def skoda(m):
    db = pd.read_csv('skoda.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def merc(m):
    db = pd.read_csv('merc.csv')

    db = db[db["model"].str.contains("SLK") == False]
    db = db[db["model"].str.contains("SL CLASS") == False]
    db = db[db["model"].str.contains("G Class") == False]
    db = db[db["model"].str.contains("GLE Class") == False]
    db = db[db["model"].str.contains("GLA Class") == False]
    db = db[db["model"].str.contains("GLC Class") == False]
    db = db[db["model"].str.contains("S Class") == False]
    db = db[db["model"].str.contains("E Class") == False]
    db = db[db["model"].str.contains("GL Class") == False]
    db = db[db["model"].str.contains("CLS Class") == False]
    db = db[db["model"].str.contains("CLC Class") == False]
    db = db[db["model"].str.contains("CLA Class") == False]
    db = db[db["model"].str.contains("V Class") == False]
    db = db[db["model"].str.contains("M Class") == False]
    db = db[db["model"].str.contains("CL Class") == False]
    db = db[db["model"].str.contains("GLS Class") == False]
    db = db[db["model"].str.contains("GLB Class") == False]
    db = db[db["model"].str.contains("X-CLASS") == False]


    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def toyota(m):
    db = pd.read_csv('toyota.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def vw(m):
    db = pd.read_csv('vw.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


def vauxhall(m):
    db = pd.read_csv('vauxhall.csv')
    X = db[['price', 'year', 'transmission', 'mileage', 'fuelType', 'mpg', 'engineSize']]
    Y = db[['model']]
    building_the_nn(X, Y, m)


# these functions take the input of the car make and then proceed to read off of the correct datasets and takes the
# correct collums to take as the
# input and the output variable


def car_make_pick():
    # this is the function that will be used to select what database and make of car that the neural network will train
    # itself on
    loop = 1
    while loop == 1:
        # this loop is used so that if the user inputs an invalid input they will be reasked to input
        try:
            car_make = int(input(
                'What make is the car that you are trying to predict (please enter the corresponding number):'
                '\n1. Ford\n2. Audi\n3. Mercedes\n4. BMW\n5. Hyundi\n6. Volkswagen\n7. Vauxhall\n8. Skoda\n9. Toyota '))
            if car_make >= 1 and car_make <= 9:
                loop = 0
            else:
                print('The input that you have entered is not valid please try again')
            # this breaks the loop so that if they enter a value too large or too small asking them to renter the input
            # with a message sayng that the input was invalid
        except:
            print('The input that you have entered is not valid please try again')
        # this try and except statement makes it so that if what they have entered causes an error it catches it and
        # spits them out back at the strat of the input
    print ('')
    if car_make == 1:
        ford(car_make)
    if car_make == 2:
        audi(car_make)
    if car_make == 3:
        merc(car_make)
    if car_make == 4:
        bmw(car_make)
    if car_make == 5:
        hyundi(car_make)
    if car_make == 6:
        vw(car_make)
    if car_make == 7:
        vauxhall(car_make)
    if car_make == 8:
        skoda(car_make)
    if car_make == 9:
        toyota(car_make)
    # these statements make it so that depending on their input that they input when inputing the progam sends them to
    # the corresponding functions

car_make_pick()

def inputs():
    price = float(input('What is the price of your car?: '))
    year = float(input('What is the year of your car?: '))
    millage = float(input('What is the millage of your car?: '))
    fuel = str(input('What is the fuel type of your car?: '))
    transmission = str(input('Is your car an automatic or manual?: '))
    mpg = float(input('What miles per gallon is your car?: '))
    engine_size = float(input('What is your engine size (in litres)?: '))
