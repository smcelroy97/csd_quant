# **csd_quant**

>These file can be used to run **PCA** on a set of **CSD** data to create an ideal case of your  
>phenomenon of interest  
>
>This ideal case can then be compared against other recordings to provide a score  
>to later be used in model optimization

### **wasserstein_distance.py**

> Loads Two CSD files to calculate the Wasserstein distance between their
> sinks and sources  
> These distances are then summed to create a total Wasserstein Distance score

### **csd_erp.py**

> WIP at the moment, meant to perform **PCA** on a directory of **CSD** data to create an ideal  
> case CSD to be used in ***wasserstein_distance.py***

### **utils.py**

> Utility functions used in the files above - to be moved into its own dir later