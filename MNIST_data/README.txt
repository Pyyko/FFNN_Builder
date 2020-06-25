----------------------------------------- Not very important -----------------------------------------


Other data than the MNIST dataset is supported
(I didn't test it with other dataset, idk if it's really working)

BUT it has to be a pickle object, a list : [image_data, label_data]

Where image_data and label_data are np.array


/!\ There is a risk of memory error with bigger dataset.


One way to fixed it would be to not save and load an entire dataset into one list 
but to iterate on the csv.

Maybe one day i will implement it... maybe 


PS : (if there is a problem with the data you can download the MNIST dataset under .csv 
and stuff.download_data will convert it into pickles object)

