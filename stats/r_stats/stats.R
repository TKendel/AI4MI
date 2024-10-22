
#install.packages("reticulate")
library(reticulate)
#install.packages("rlist")
library(rlist)
#install.packages("car")
library(car)
#install.packages("dgof")
#library(dgof)
#install.packages("dgtidyverseof")
#library(tidyverse)
#install.packages("plyr")
library(plyr)



np <- import("numpy")

# 
# #Best epoch 53 -- which is 54 on R
# test = np$load('FINALBASELINE/3ddice_val.npy')
# print(test)
# 
# best_test = test[54]
# print(best_test)
# 
# dim(test)
# 
# best_test2 =  test[54,1,1]
# print(best_test2)
# best_test2 =  test[54,1,2]
# print(best_test2)
# 
# 
# matrix <- test[54, , ]
# print(matrix)
# 
# 
# 
# 
# 
# test2 = np$load('results/ERG3C2B/3ddice_val.npy')
# print(test2)
# matrix2 <- test2[51, , ]
# print(matrix2)
# 
# row1_class2 = matrix[,2]
# row2_class2 = matrix2[,2]
# print(row1_class2)
# 
# hist(row1_class2)
# hist(row2_class2)
# erg3c2b=c('ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B','ERG3C2B')
# baseline=c('BASELINE','BASELINE','BASELINE','BASELINE','BASELINE','BASELINE','BASELINE','BASELINE','BASELINE','BASELINE')
# 
# main = data.frame(tag=c(erg3c2b,baseline),dice=c(row1_class2, row2_class2))
# print(main)
# 
# t.test(dice ~ tag, data = main)
# 
# 
# t.test(row1_class2,row2_class2, alternative = "greater")
# 







length_warning = list()

#for i in lenth best epochs (one per folder)
file_list = list('3ddice_val.npy', '3dIOU_val.npy', '95HD_val.npy', 'ASD_val.npy', 'cldice_val.npy', 'dice_val.npy', 'HD_val.npy', 'iou_val.npy') #files not used 'closs_val.npy', 'loss_val.npy'

#path = paste("results/FINALBASELINE/", file_list[1], 'other', sep="")

preprocessing = list('BER','EAR', 'ERG', 'ERG3C2B' )
best_epoch = list(40, 43, 23, 20 )
for(p in 1:length(preprocessing)){
  
  results_t = list()
  results_lavene = list()
  results_ks = list()
  #rs = data.frame(class=c('background','class2','class3','class4','class5'))
  for (x in 1:length(file_list)) {
    filename=file_list[x]
    print(paste('File',  filename))
    base_log = np$load(paste("results/FINALBASELINE/", filename, sep=""))
    #print(paste('dimentions',dim(base_log)))
    base_matrix = base_log[44, , ] # 43python, 44 in R, From the best_epoch file
    #tag='base'
    epoch = unlist(best_epoch[p])
    print(epoch)
    BER_log = np$load(paste("results/", preprocessing[p], '/', filename, sep="")) #np$load('results/BER/3ddice_val.npy')
    BER_matrix = BER_log[epoch, , ] # 39python, 40R From the best_epoch file
    #tag='BER'
    
    shape = dim(BER_matrix)
    #print(unlist(filename))
    #Remove from string
    #https://sparkbyexamples.com/r-programming/remove-character-from-string-in-r/
    filename = list(gsub('[_val.npy]','',unlist(filename)))
    p_values = list(filename)
    p_lavene = list(filename)
    p_ks = list(filename)
    #p_ = list()
    if(length( base_matrix[,1])>10){
      length_warning[[length(length_warning)+1]] <- paste('Warning', filename, 'of length', length(base_class))
    }
    for (i in 1:shape[2]){
      base_class =  base_matrix[,i]
      BER_class =  BER_matrix[,i]
      test = t.test(base_class,BER_class, alternative = "greater")
      ks_test = ks.test(base_class, BER_class)
      #df fot levene
      print(summary(ks_test))
      print(ks_test$p.value)
      
      print(paste('HEREEEER',length(base_class),length(BER_class)))
      #print(base_class)
      tags = c(rep('baseline', length(base_class)), rep(preprocessing[p], length(BER_class))) #10 because there is 10 entries on each validation - make resizable
      metrics = c(base_class,BER_class)
      df_levene = data.frame(tag=tags, metric=metrics)
      print(length(metrics))
      print(length(tags))
      levene=leveneTest(metrics ~ unlist(tags))
      #levene[1,3]#p value

      #print(test$p.value)
      p_values = c(p_values, signif(test$p.value, digits = 4))
      p_lavene = c(p_lavene,  signif(levene[1,3], digits = 4))
      p_ks = c(p_ks, signif(ks_test$p.value, digits = 4) )
      #p_ = c(p_, test$p.value)
      #rs[filename][[i,1]]=test$p.value
      
    }
    #Because of formatting of the data, this needs to be hard coded
    if(length(p_values)==5){
      #No background case
      p_values=append(p_values, "Null", after = 1)
      p_lavene=append(p_lavene, "Null", after = 1)
      p_ks=append(p_ks, "Null", after = 1)
    }
    if(length(p_values)==3){
      #No background case
      p_values=append(p_values, "Null", after = 1)
      p_lavene=append(p_lavene, "Null", after = 1)
      p_ks=append(p_ks, "Null", after = 1)
      for (r in 1:2){
        p_values=append(p_values, "Null", after = 3)
        p_lavene=append(p_lavene, "Null", after = 3)
        p_ks=append(p_ks, "Null", after = 3)
      }
    }
    if (length(p_values)<6){
      difference = 6-length(p_values)
      fill_in = rep('NULL', difference)
      p_values = c(p_values, fill_in)
      p_lavene = c(p_lavene, fill_in)
      p_ks = c(p_ks, fill_in)

    }
    
    
    # 
    # print(p_ks)
    # dff = data.frame(p_ks)
    # dff
    #print(p_values)
    #length(p_values)
    #results = append(results, p_values)
    results_t[[length(results_t)+1]] <- p_values
    results_lavene[[length(results_lavene)+1]] <- p_lavene
    results_ks[[length(results_ks)+1]] <- p_ks
    #results_p[[length(results_ks)+1]] <- p_
    
    #print(unlist(p_))
    #rs[filename]=unlist(p_)
    #temp = data.frame(p_)
  }
  colum_names = c('Metric', 'Background', 'Esophagus', 'Heart', 'Trachea','Aorta')
  results_t <- plyr::adply(results_t,1,unlist,.id = NULL)
  colnames(results_t) = colum_names
  results_lavene <- plyr::adply(results_lavene,1,unlist,.id = NULL)
  colnames(results_lavene) = colum_names
  results_ks <- plyr::adply(results_ks,1,unlist,.id = NULL)
  colnames(results_ks) = colum_names
  #https://stackoverflow.com/questions/47876074/how-to-write-a-list-of-lists-into-a-single-csv-file-in-r

  #print(results_t)
  filepath = paste("p_values/", preprocessing[p],'.t_test.csv', sep="")
  #capture.output(results_t, file=filepath)
  write.csv(results_t,filepath)
  
  filepath = paste("p_values/", preprocessing[p],'.levene.csv', sep="")
  #capture.output(results_lavene, file=filepath)
  write.csv(results_lavene,filepath)
  
  filepath = paste("p_values/", preprocessing[p],'.kolmogorov.csv', sep="")
  #capture.output(results_ks, file=filepath)
  write.csv(results_ks,filepath) #*****
  
  print(paste('Done with', preprocessing))
  
  #print(results_p)
  
  
  

  # print(results_t)
  # new_list <- plyr::adply(results_t,1,unlist,.id = NULL)
  # new_list
  # #https://stackoverflow.com/questions/47876074/how-to-write-a-list-of-lists-into-a-single-csv-file-in-r
  # 
  # > new_list
  # 
  # V1 V2 V3
  # 1 aa bb cc
  # 2 xx yy zz
  # 
  # > write.csv(new_list, "mycsv.csv")

  # 
  # print(results_t)
  # print(results_t[1])
  # print(unlist(results_t[1]))
  # line =results_t[1]
  # #print(line[[1]][[4]][1])#inditem
  # #print(line[[1]][-1])#remove first item
  # title=line[[1]][[1]][1]
  # print(title)
  # numbers=line[[1]][-1]
  # print(numbers)
  # df = data.frame(numbers)
  # df
  # #write.table( <yourdf>, sep=",",  col.names=FALSE)
  # 
  # capture.output(unlist(df), file='test.csv')
  # 
  # 
  # 
  # 
  # # Unnest the list of lists using unnest_longer() 
  # unnest_list <- unnest_longer(tbl, x) 
  # 
  # # Print the unnested list 
  # print(unnest_list) 
  # 
  
}


print(length_warning)
print(length(length_warning))




# 
# 
# 
# 
# 
# 
# test = np$load('FINALBASELINE/3ddice_val.npy')
# dim(test)
# 
# test = np$load('FINALBASELINE/dice_val.npy')
# dim(test)
# 
# 
# 
# 
# 
# 
# 
# df = data.frame(dice3d=results[1],dice=results[2])
# print(df)
# 
# dd  <-  as.data.frame(matrix(unlist(results), nrow=length(unlist(results[1]))))
# dim(dd)
# print(dd)
# 
# df = data.frame(results)
# df
# dim(df)
# 
# 
# 
# 
# #write_csv(results, filepath)
# #saveRDS(results, file=filepath )
# #list.save(results, filepath)
# 
# 
# print(unlist(results))
# df = data.frame(results)
# print(df)
# dim(df)
# 
# example = data.frame(title=c(1,2,3,4), beta=c(5,6,7,8))
# print(example)
# example['title'][[2,1]]#access value 
# 
# for (i in 1:shape[2]){
#   print(i)
#   
#   
# }

