import numpy as np

def gaussian_partial(A, b):
    
    n = A.shape[0]
    
    C=np.c_[A,b.reshape(-1,1)]
    
    flag = 0
    
    #column
    for i in range(n-1):
            
        max_c, chosen_k  = 0, i
        
        #find the pivot
        for k in range(i, n):
            if np.abs(C[k,i]) > max_c:
                max_c = np.abs(C[k,i])
                chosen_k = k
         
        #Check if the pivot is zero
        if max_c == 0:
            flag = 1
            break
        
        #interchange rows
        if chosen_k != i:
            #Swap 2 rows
            temp = C[i,:].copy()
            C[i,:] = C[chosen_k,:]
            C[chosen_k,:] = temp
    
        
        #row
        for j in range(i+1, n):
            
            c = C[j,i]/C[i,i]
            C[j,:] = C[j,:] - c*C[i,:]
            
    return C, flag


def backsubstitution(T):
    
    flag=0
    n = T.shape[0]
    X = np.zeros((n))
    if T[n-1,n-1] == 0:
        flag = 1
    
    else:
    
        X[n-1] = T[n-1,n]/T[n-1,n-1] 

        for i in range(n-2,-1,-1):
            #Sum in row i
            s = 0
            for j in range(i+1, n):
                s += T[i,j]*X[j]

            X[i] = (T[i,n] - s)/T[i,i]
    
    return X, flag


A = np.array([[2, 1, 5],
            [4, 4, -4],
            [1, 3, 1]])
b= np.array([8,4,5])


T, err = gaussian_partial(A,b)

if err:
    print('Not unique solution')
else:
    X, err = backsubstitution(T)
    if err:
        print('Not unique solution')
    else:
        print('Solution:', X)


np.linalg.solve(A,b)