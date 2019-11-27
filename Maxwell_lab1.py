## create a list 'things' that contains three things: mozarella, cinderella, salmonella
## print the list  
print('\n\nCreate a list "things" that contains three things: mozarella, cinderella, salmonella. Print the list:\n')
test = ['mozarella', 'cinderella', 'salmonella']
print(test)

## Capitalize the thing in the list that refers to a person.  Print the list
print('\n\nCapitalize the thing in the list that refers to a person.  Print the list:\n')
test[1] = 'Cinderella'
print(test)

## Put the cheese-thing in all upercase and print the list.
print('\n\nPut the cheese-thing in all upercase and print the list:\n')
test[0] = 'MOZARELLA'
print(test)

## delete the disease element from the list.  Print the list.
print('\n\ndelete the disease element from the list.  Print the list:\n')
test.remove('salmonella')
print(test)

## Create a list `movies` that contains the names of six movies you like
## Using a loop, print out all the movies
print('\n\nCreate a list `movies` that contains the names of six movies you like. Using a loop, print out all the movies:\n')
movies = ['movie A', "movie B", "movie C", "movie D", "movie E", "movie F"]
for x in movies:
    print(x)

## Create a list `top_three` that only contains the first three movies in the list.
## Use indexing and slicing!
print('\n\nCreate a list `top_three` that only contains the first three movies in the list:\n')
top_three = movies[0:3]
print(top_three)

## Using a loop, create a new list `excited` that adds the phrase -
## " is a great movie!" to the end of each element in your movies list
print('\n\nUsing a loop, create a new list `excited` that adds the phrase - " is a great movie!" to the end of each element in your movies list:\n')
excited = []
for x in movies:
    excited.append(x + " is a great movie!")
print(excited)

## Create a list `numbers` that is the numbers 70 through 99
## note: `range()` does not give you a list.  Use a loop.
print('\n\nCreate a list `numbers` that is the numbers 70 through 99:\n')
numbers = []
for x in range(70,100):
    numbers.append(x)
print(numbers)

## Using the built in len function, create a variable `length` that is equal to the length of your list
## `numbers
print('\n\nUsing the built in len function, create a variable `length` that is equal to the length of your list `numbers`:\n')
length = len(numbers)

## Use a loop to compute the mean value of the list.
print('\n\nUse a loop to compute the mean value of the list:\n')
total = 0
for i in numbers:
    total += i
mean = total / length 
print(mean)

## create a list of 20 numbers, and extract the odd numbers using slicing
print('\n\ncreate a list of 20 numbers, and extract the odd numbers using slicing:\n)')
test = []
for x in range(50,71):
    test.append(x)

odd = test[1::2]
print(odd)

###DICTS

## create a dict that links countries to their capitals (at least 5 countries)
print('\n\ncreate a dict that links countries to their capitals (at least 5 countries)):\n')
test = {'Korea': 'Seoul', 'Thailand':'Bangkok','Jerusalem':'Israel', 'Paris':'France', 'London':'Britain'}
for key in test.keys():
    print('The capital of ' + key + ' is ' + test.get(key))


