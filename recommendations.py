'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Aubrey Parks

'''

import os
import math 
import matplotlib.pyplot as plt
import numpy as np 
import pickle 

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile) as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def stdev(lst, avg): 
    tot = 0
    for i in lst: 
        tot+= (i -avg)**2
    tot = tot/len(lst)
    tot = math.sqrt(tot)
    return tot

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    names = prefs.keys()
    # create lists containing the total items, the items once, and the total 
    #ratings 
    totItems = []
    items = []
    totRatings = []
    for i in names: 
        temp = prefs[i]
        for j in temp.keys(): 
            totItems.append(j)
            totRatings.append(temp[j])
            if j not in items: 
                items.append(j)
                
    print('Stats for: %s' % filename) 
    print('Number of users: %d' % len(names))
    print('Number of items: %d' % len(items))
    print('Number of ratings: %d' % len(totItems))
            
    #determine overall average rating and stdev
    totAvg = 0
    for i in totRatings: 
        totAvg += i
    totAvg = totAvg/len(totItems)
    
    totStdev = stdev(totRatings, totAvg)
    
    print('Overall average rating: %.2f out of 5, and std of %.2f' %(totAvg,totStdev))
    
    # determine the average item rating 
    counter = 0
    itemRatings = []
    for i in items: 
        totRat = 0
        itemCounter = 0
        for j in totItems: 
            if i==j: 
                itemCounter+=1
                totRat += totRatings[counter]
            counter+=1
        totRat = totRat/itemCounter
        itemRatings.append(totRat)
        counter = 0
        
    itemAvg = 0
    for i in itemRatings: 
        itemAvg += i
    itemAvg = itemAvg/len(itemRatings)
    itemStdev = stdev(itemRatings, itemAvg)
    print('Average item rating: %.2f out of 5, and std of %.2f' %(itemAvg, itemStdev))
    
    #determine the average user rating and stdev 
    userRat = []
    for i in names: 
        temp = prefs[i]
        userTot = 0
        numMovies = 0
        for j in temp.keys(): 
            userTot += temp[j]
            numMovies +=1
        userTot = userTot/numMovies
        userRat.append(userTot)
    userAvg = 0
    for i in userRat: 
        userAvg += i
    userAvg = userAvg/len(userRat)
    userStdev = stdev(userRat, userAvg)
    print('Average user rating: %.2f out of 5, and std of %.2f' %(userAvg, userStdev))
    
    #determine sparsity 
    sparsity = (1 - (len(totRatings)/(len(names)*len(items))))*100
    sparsity = round(sparsity, 2)
    print('User-Item Matrix Sparsity: ' + str(sparsity) + "%")
    #create the histogram 
    plt.hist(totRatings, bins = [1, 2, 3, 4, 5], color='g')
    plt.title("Ratings Histogram")
    plt.xlabel("Rating")
    plt.ylabel("Number of User ratings")
    plt.xticks([1,2,3,4,5])
    plt.yticks([0,2,4,6,8,10,12,14,16,18])
    plt.show()
    
    return 

def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    names = prefs.keys()
    movies = {}
    # create a dictionary with the ratings organized by movie
    for i in names: 
        temp = prefs[i]
        for i in temp.keys(): 
            if i in movies:
                lst = movies[i]
                lst.append(temp[i])
                movies[i] = lst
            else: 
                movies[i] = [temp[i]]
     # update the diction to have the average rating and number organized by 
     #the movie            
    for i in movies.keys(): 
        lst = movies[i]
        avg = 0
        for j in lst: 
            avg += j
        avg = avg/len(lst)
        newLst = [avg, len(lst)]
        movies[i] = newLst

    #sort the dictionary by number of ratings and print information 
    sortedRating = sorted(movies.items(), key=lambda kv:kv[1][1])
    sortedRatingDict = dict(sortedRating)
    print()
    print('Popular items -- most rated:')
    print('Title \t\t\t\t\t #Ratings \t Avg Rating')
    movieKey = list(sortedRatingDict.keys())
    for i in range(5, 0, -1): 
        print('%s \t\t\t %d \t\t %.2f' %(str(movieKey[i]), sortedRatingDict[movieKey[i]][1], sortedRatingDict[movieKey[i]][0]))
    # sort the dictionary by rating and print information 
    print()
    print('Popular items -- highest rated:')
    print('Title \t\t\t\t\t Avg Rating \t #Ratings')
    sortedHighest = sorted(movies.items(), key=lambda kv:kv[1])
    sortedHighestDict = dict(sortedHighest)
    highestKey = list(sortedHighestDict.keys())
    
    for i in range(5, 0, -1):
        print('%s \t\t\t %.2f \t\t %d' %(str(highestKey[i]), sortedHighestDict[highestKey[i]][0], sortedHighestDict[highestKey[i]][1]))
    #print information for overall best rated items 
    print()
    print('Overall best rated items (number of ratings >= 5):')
    print('Title \t\t\t\t\t Avg Rating \t #Ratings')
    for i in range(5, 0, -1): 
        if(sortedHighestDict[highestKey[i]][1] >= 5): 
            print('%s \t\t\t %.2f \t\t %d' %(str(highestKey[i]), sortedHighestDict[highestKey[i]][0], sortedHighestDict[highestKey[i]][1]))
    return 

# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity for RS, as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    person1List = prefs[person1]
    person2List = prefs[person2]
    sum_of_squares = 0
    
    for i in si: 
        sum_of_squares += (person1List[i] - person2List[i])**2

    # returns Euclidean distance similarity for RS
    return 1/(1+math.sqrt(sum_of_squares)) 

def sim_pearson(prefs,p1,p2):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
    lst1 = prefs[p1]
    lst2 = prefs[p2]
    same = []
    for i in lst1.keys(): #finds the similarities between people
        if i in lst2: 
            same.append(i)

    rat1 = []
    rat2 = []
    for i in same: #adds all the similar ratings to lists 
        rat1.append(lst1[i])
        rat2.append(lst2[i])
        
    avg1 = np.average(rat1)
    avg2 = np.average(rat2)
    denom1 = 0
    denom2 = 0
    numerator = 0
    
    #for every value in the similar lists, find the numerator and denominator
    for i in same: 
        numerator += (lst1[i] - avg1)*(lst2[i] - avg2)
        denom1 += (lst1[i] - avg1)**2
        denom2 += (lst2[i] - avg2)**2
    denom1 = math.sqrt(denom1)
    denom2 = math.sqrt(denom2)
    denominator = denom1*denom2
    if denominator == 0: #avoid a divide by zero issue
        return -1
    
    return numerator/denominator

## add tbis function to the other set of functions
# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other==person: 
            continue
        sim=similarity(prefs,person,other)
    
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings


def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset
                
    Parameters
        -- prefs: nested dictionary containing a U-I matrix
        -- sim: similarity function to use (default = sim_pearson)
        -- num_users: max number of users to print (default = 10)
        -- top_N: max number of recommendations to print per user (default = 5)
        
    Returns: None
    '''
    names = prefs.keys() #determine the names 
    if len(names) > num_users: 
        names = names[0:num_users]
    
    if sim == sim_pearson: 
        for i in names: # find the ranking for every name for the right number 
            rankings = getRecommendations(prefs, i)
            print('User-based CF recs for %s: ' %(i), rankings[0:top_N+1])
            
    if sim == sim_distance: 
        for i in names: #find the ranking for every name for the right number 
            rankings = getRecommendations(prefs, i, similarity=sim_distance)
            print("User-based CF recs for %s: " %(i), rankings[0:top_N+1])
            
            
def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    """
    temp = prefs
    errorList = []
    meanError = 0
    for name in prefs.keys(): #for every name
        lst = list(temp[name].keys()) #create a list of movies 
        for i in range(len(lst)): #for every movie 
            value = prefs[name][lst[i]] #save the rating 
            del temp[name][lst[i]] #remove the movie from temp 
            if sim == "sim_pearson": #find the recommendation 
                results = getRecommendations(temp, name)
            else: 
                results = getRecommendations(temp, name, sim_distance)
            temp[name][lst[i]] = value #replace the deleated values 
            for j in range(len(results)): 
                if results[j][1] == lst[i]: #if the recommendation contains the movie 
                    error = (results[j][0] - prefs[name][lst[i]])**2
                    errorList.append(error) #determine the error and print infomration 
                    meanError += error
                    print("User: %s, item: %s, Prediction: %f, Actual: %f, Sq Error: %f" %(name, lst[i], results[j][0], value, error))
    if metric == "MSE": 
        N = len(errorList) #calculate the mean square error 
        meanError = meanError/N
        print("%s for critics: %f using <function %s at 0x7fba06122940>" %(metric, meanError, sim))

    return meanError, errorList
            

def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=10,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d") % (c,len(itemPrefs))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item] = scores
    return result

def getRecommendedItems(prefs,itemMatch,user):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    '''
    for i in prefs.keys(): 
        user_name = i 
        print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                       getRecommendedItems(prefs, itemsim, user_name)) 

def loo_cv_sim(prefs, metric, sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
	 metric: MSE, or MAE, or RMSE
	 sim: distance, pearson, etc.
	 algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
	 
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
	 error_list: list of actual-predicted differences
    """
    print("Number of users processed: 0")
    error_list = []
    error_total  = 0
    temp = prefs # create a temp list 
    for name in prefs.keys(): 
        lst = list(temp[name].keys()) #create a list of items 
        for i in range(len(lst)): 
            ans = False #create boolean to determine if has recommendation 
            value = prefs[name][lst[i]]
            del temp[name][lst[i]]# save value and delete from temp 
            result = getRecommendedItems(temp, sim_matrix, name) #get recommendations 
            temp[name][lst[i]] = value #replace the value 
            for j in range(len(result)): 
                if result[j][1] == lst[i]: #if the result matches the item 
                    error = 0 #calculate the error based on instruction 
                    if metric == 'MSE' or metric == 'RMSE': 
                        error = (result[j][0] - prefs[name][lst[i]])**2
                    if metric == 'MAE': 
                        error = abs(result[j][0] - prefs[name][lst[i]])
                    error_list.append(error)
                    error_total += error 
                    print("User: %s, item: %s, Prediction: %f, Actual: %f, Error: %f" %(name, lst[i], result[j][0], prefs[name][lst[i]], error))
                    ans = True #set boolean to true 
            if ans == False: #if not in the result list 
                print("From loo_cv_sim(), No prediction calculated for item %s, user %s in pred_list: %s" %(lst[i], name, str(result)))

    N = len(error_list) #calculate final error
    error_total = error_total/N
    if metric == "RMSE": 
        error_total = np.sqrt(error_total)
    return error_total, error_list
        
def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    #path = path + '/Desktop/starter' #need to run correct path on my laptop
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print? \n'
                        'D(istance) critics data? \n'
                        'PC(earson Correlation) critics data? \n'
                        'U(ser-based CF Recommendations)?\n'
                        'LCV(eave one out cross-validation)?\n'
                        'LCVSIM(eave one out cross-validation)?\n'
                        'Sim(ilarity matrix) calc for Item-based recommender?\n'
                        'I(tem-based CF Recommendations)?, \n==> ')
                        
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))
            
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')   
                
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                print('Examples:')
                print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                num=1
                den=(1+ math.sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                print('Distance sim Lisa & Gene (check):', num/den)    
                print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                print()
                
                print('User-User distance similarities:')
                
                names = prefs.keys() 
                for i in names: 
                    for j in names: 
                        if i != j: 
                            dist = sim_distance(prefs, i, j)
                            print('Distance sim %s & %s: %f' %(i, j, dist))
                print()
            else:
                print ('Empty dictionary, R(ead) in some data!')  
                
        # Testing the code ..
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')
                # Calc Pearson for all users
                
                names = prefs.keys()
                for i in names: 
                    for j in names: 
                        if i != j: 
                            pearson = sim_pearson(prefs, i, j)
                            print("Pearson sim %s & %s: %f" %(i, j, pearson))
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
                
        elif file_io == 'U' or file_io == 'u':

            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendations(prefs, user_name)) 
                        # [(3.3477895267131017, 'The Night Listener'), 
                        #  (2.8325499182641614, 'Lady in the Water'), 
                        #  (2.530980703765565, 'Just My Luck')]
                print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                       getRecommendations(prefs, user_name, similarity=sim_distance)) 
                        # [(3.457128694491423, 'The Night Listener'), 
                        #  (2.778584003814924, 'Lady in the Water'), 
                        #  (2.422482042361917, 'Just My Luck')]
                print()
                
                print('User-based CF recommendations for all users:')
                # Calc User-based CF recommendations for all users
                print("Using sim_pearson:")
                get_all_UU_recs(prefs)
                print()
                print("Using sim_distance:")
                get_all_UU_recs(prefs, sim=sim_distance)

                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
                
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
                print ('Example:')            
                ## add some code here to calc LOOCV 
                ## write a new function to do this ..
                print("LOO_CV Evaluation")
                loo_cv(prefs, "MSE", "sim_pearson", "user")
                print()
                loo_cv(prefs, "MSE", 'sim_distance', "user")
            else:
                print ('Empty dictionary, R(ead) in some data!')   

        elif file_io == 'Sim' or file_io == 'sim':
            
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                
                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    movies = itemsim.keys()
                    for i in movies: 
                        print(i, end='')
                        print(itemsim[i])
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:                
                print ('Example:')
                user_name = 'Toby'
    
                print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                       getRecommendedItems(prefs, itemsim, user_name)) 
                
                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_distance:  
                ##     [(3.1667425234070894, 'The Night Listener'), 
                ##      (2.9366294028444346, 'Just My Luck'), 
                ##      (2.868767392626467, 'Lady in the Water')]
                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_pearson:  
                ##     [(3.610031066802183, 'Lady in the Water')]
                ##
    
                print()
                
                print('Item-based CF recommendations for all users:')
                # Calc Item-based CF recommendations for all users
        
                ## add some code above main() to calc Item-based CF recommendations 
                ## ==> write a new function to do this, as follows
                    
                get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                # Note that the item_sim dictionry and the sim_method string are
                #   setup in the main() Sim command
                
                ## Expected Results ..
                
                ## Item-based CF recs for all users, sim_distance:  
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_distance:  []
                ## Item-based CF recs for Gene, sim_distance:  []
                ## Item-based CF recs for Michael, sim_distance:  [(3.2059731906295044, 'Just My Luck'), (3.1471787551061103, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_distance:  [(3.43454674373048, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_distance:  []
                ## Item-based CF recs for Jack, sim_distance:  [(3.5810970647618663, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_distance:  [(3.1667425234070894, 'The Night Listener'), (2.9366294028444346, 'Just My Luck'), (2.868767392626467, 'Lady in the Water')]
                ##
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_pearson:  []
                ## Item-based CF recs for Gene, sim_pearson:  []
                ## Item-based CF recs for Michael, sim_pearson:  [(4.0, 'Just My Luck'), (3.1637361366111816, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_pearson:  [(3.4436241497684494, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_pearson:  []
                ## Item-based CF recs for Jack, sim_pearson:  [(3.0, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_pearson:  [(3.610031066802183, 'Lady in the Water')]
                    
                print()
                
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    

        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            if len(prefs) > 0 and itemsim !={}:             
                print('LOO_CV_SIM Evaluation')
                if len(prefs) == 7:
                    prefs_name = 'critics'

                metric = input ('Enter error metric: MSE, MAE, RMSE: ')
                if metric == 'MSE' or metric == 'MAE' or metric == 'RMSE' or \
		        metric == 'mse' or metric == 'mae' or metric == 'rmse':
                    metric = metric.upper()
                else:
                    metric = 'MSE'
                algo = getRecommendedItems ## Item-based recommendation
                
                if sim_method == 'sim_pearson': 
                    sim = sim_pearson
                    error_total, error_list  = loo_cv_sim(prefs, metric, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s' 
			  % (metric, prefs_name, error_total, len(error_list), sim) )
                    print()
                elif sim_method == 'sim_distance':
                    sim = sim_distance
                    error_total, error_list  = loo_cv_sim(prefs, metric, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s' 
			  % (metric, prefs_name, error_total, len(error_list), sim) )
                    print()
                else:
                    print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                if prefs_name == 'critics':
                    print(error_list)
            else:
                print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')

        else: 
            done = True
                    
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()