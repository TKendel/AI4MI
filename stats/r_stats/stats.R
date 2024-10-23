
#install.packages("reticulate")
library(reticulate)
#install.packages("rlist")
library(rlist)
#install.packages("car")
library(car)
#install.packages("plyr")
library(plyr)



np <- import("numpy")

length_warning = list() #This can be ignored, some files were showing length beyond the expected
                        #those files are not created here, so it is handled, but not prevented
                        #This just recopilates cases in which this is true

#File list
file_list = list('3ddice_val.npy', '3dIOU_val.npy', '95HD_val.npy', 'ASD_val.npy', 'cldice_val.npy', 'dice_val.npy', 'HD_val.npy', 'iou_val.npy') 
#Types of processioning and thier best epoch, list length must match
preprocessing = list('BER','EAR', 'ERG', 'ERG3C2B' )
best_epoch = list(40, 43, 23, 20 )
for(p in 1:length(preprocessing)){
  #Holders for results of every test
  results_t = list()
  results_lavene = list()
  results_ks = list()
  for (x in 1:length(file_list)) {
    filename=file_list[x]
    print(paste('File',  filename))
    #BASELINE
    base_log = np$load(paste("results/FINALBASELINE/", filename, sep=""))#load matrix
    base_matrix = base_log[44, , ] # 43python, 44 in R, From the best_epoch file
    #this above gathers the 2d matrix of the best epoch; 44 is fixed for the baseline
    #PREPROCESSED
    epoch = unlist(best_epoch[p])
    BER_log = np$load(paste("results/", preprocessing[p], '/', filename, sep="")) #np$load('results/BER/3ddice_val.npy')
    BER_matrix = BER_log[epoch, , ] #Gatehr matrix for best epoch. (e.g. 39 in python, 40 in R) from the best_epoch file
    
    shape = dim(BER_matrix)
    #Remove from string
    #https://sparkbyexamples.com/r-programming/remove-character-from-string-in-r/
    filename = list(gsub('[_val.npy]','',unlist(filename))) #remove _val.npy for cleaner naming
    p_values = list(filename)#p value holder
    p_lavene = list(filename)
    p_ks = list(filename)
    if(length( base_matrix[,1])>10){
      #This compiles warnings for files that are larger than expected, no effect takes place and
      # the files are handled regardless of their length
      length_warning[[length(length_warning)+1]] <- paste('Warning',  preprocessing[p], '-',filename, 'of length', length( base_matrix[,1]))
    }
    for (i in 1:shape[2]){
      #Extract array for the specific class, from baseline and reprocessing
      base_class =  base_matrix[,i]
      BER_class =  BER_matrix[,i]
      #Apply t test - we are only concerned if the mean is different
      test = t.test(base_class,BER_class, alternative = "greater")
      #Apply Kolmogorov-Smirnov test - H0 the two dataset values are from the same continuous distribution
      ks_test = ks.test(base_class, BER_class)
      #Data preparisson for Levene test
      tags = c(rep('baseline', length(base_class)), rep(preprocessing[p], length(BER_class))) #10 because there is 10 entries on each validation - make resizable
      metrics = c(base_class,BER_class)
      levene=leveneTest(metrics ~ unlist(tags))
      #Save p values
      p_values = c(p_values, signif(test$p.value, digits = 4))
      p_lavene = c(p_lavene,  signif(levene[1,3], digits = 4))
      p_ks = c(p_ks, signif(ks_test$p.value, digits = 4) )
    }
    #Because of formatting of the data, this needs to be hard coded
    if(length(p_values)==5){
      #No background case (2nd position)
      p_values=append(p_values, "N/A", after = 1)
      p_lavene=append(p_lavene, "N/A", after = 1)
      p_ks=append(p_ks, "N/A", after = 1)
    }
    if(length(p_values)==3){
      #clDice case, no second, forth nor fifth value
      p_values=append(p_values, "N/A", after = 1)
      p_lavene=append(p_lavene, "N/A", after = 1)
      p_ks=append(p_ks, "N/A", after = 1)
      for (r in 1:2){
        p_values=append(p_values, "N/A", after = 3)
        p_lavene=append(p_lavene, "N/A", after = 3)
        p_ks=append(p_ks, "N/A", after = 3)
      }
    }
    #In case any is missed (or has missing values), avid error for difference in length
    if (length(p_values)<6){
      difference = 6-length(p_values)
      fill_in = rep('N/A', difference)
      p_values = c(p_values, fill_in)
      p_lavene = c(p_lavene, fill_in)
      p_ks = c(p_ks, fill_in)
    }
    #Save lists of p values into list of lists (didn't use a DF because earlier I was handling lists of different length)
    results_t[[length(results_t)+1]] <- p_values
    results_lavene[[length(results_lavene)+1]] <- p_lavene
    results_ks[[length(results_ks)+1]] <- p_ks
  }
  #Entries, naming
  colum_names = c('Metric', 'Background', 'Esophagus', 'Heart', 'Trachea','Aorta')
  results_t <- plyr::adply(results_t,1,unlist,.id = NULL) #Reformat list of lists into table
  colnames(results_t) = colum_names#rename columns
  results_lavene <- plyr::adply(results_lavene,1,unlist,.id = NULL)
  colnames(results_lavene) = colum_names
  results_ks <- plyr::adply(results_ks,1,unlist,.id = NULL)
  colnames(results_ks) = colum_names
  #https://stackoverflow.com/questions/47876074/how-to-write-a-list-of-lists-into-a-single-csv-file-in-r

  #Save results to file
  filepath = paste("p_values/", preprocessing[p],'.t_test.csv', sep="")
  #capture.output(results_t, file=filepath)
  write.csv(results_t,filepath, quote=FALSE)
  
  filepath = paste("p_values/", preprocessing[p],'.levene.csv', sep="")
  #capture.output(results_lavene, file=filepath)
  write.csv(results_lavene,filepath, quote=FALSE)
  
  filepath = paste("p_values/", preprocessing[p],'.kolmogorov.csv', sep="")
  #capture.output(results_ks, file=filepath)
  write.csv(results_ks,filepath, quote=FALSE)
  
  print(paste('Done with', preprocessing))
}



#Print warnings - optional
print(length_warning)
print(paste('Total warnings:', length(length_warning)))



