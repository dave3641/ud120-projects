#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print "\nNumber of people in the Enron dataset: ", len(enron_data)

print "\nNumber of features for each person: ", len(enron_data.items()[0])

pois = [names for names, feature in enron_data.items() if feature['poi']]
print "\nNumber of POI's: ", len(pois)
#print sum(1 for p in enron_data if p.items('poi') == 'True')
#sum(p.poi == 'True' for p in enron_data)#
#print enron_data