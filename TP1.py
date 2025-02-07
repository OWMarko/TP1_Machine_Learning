import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

############
#Exercice 1

#Question 1, voir feuille manuscrite

#Question 2
def noise(sigma):
    if sigma < 0:
        raise ValueError("Sigma doit être positif")
    U1 = np.random.rand()
    U2 = np.random.rand()
    b = sigma * np.sqrt(-2*np.log(U1))* np.cos(2*np.pi *U2)
    return b

#Question 3
def F(x,y,w,alpha):
    if len(x) != len(y):
        raise ValueError("Erreur taille")
    N = len(x)
    i = 0
    for j in range(N):
        i = (abs(y[j] - w*x[j] -alpha))**2 + i
    return i

#Question 4
def generationWalpha():
    w = np.random.uniform(-1, 1)
    alpha = np.random.uniform(-1, 1)
    return w, alpha

#Question 5
def generationx(n):
    x = np.random.uniform(-1, 1, n)
    return x

#Question 6
def generationy(x, sigma):
    w, alpha = generationWalpha()
    bruit = np.array([noise(sigma) for _ in range(len(x))])
    y = w * np.array(x) + alpha + bruit
    return y.tolist()

#Question 7 : feuille manuscrite

#Question 8 : feuille manuscrite

#Question 9
def regression(x, y):
    xi = np.array(x)
    yi = np.array(y)
    x_moyenne = np.mean(xi)
    y_moyenne = np.mean(yi)
    diffx = xi - x_moyenne
    diffy = yi - y_moyenne
    w = np.sum(diffx*diffy) / np.sum(diffx**2)
    alpha = y_moyenne - w *x_moyenne
    return w, alpha

#Question 10
def comparaison(x, sigma):
    erreur = 1e-9
    y = generationy(x, sigma)
    w, alpha = regression(x, y)
    x_np = np.array(x).reshape(-1, 1)
    y_np = np.array(y)
    reg = LinearRegression().fit(x_np, y_np)
    w_sk = reg.coef_[0]
    alpha_sk = reg.intercept_
    if abs(w_sk - w) < erreur and abs(alpha - alpha_sk) < erreur:
        print(f"Même valeurs.")
        print(f"Biais de notre méthode : {alpha} et biais avec le module sklearn : {alpha_sk}")
        print(f"Poids de notre méthode : {w} et poids avec le module sklearn : {w_sk}")
    else:
        print("Valeurs différentes (légèrement).")
        print(f"Biais de notre méthode : {alpha} et biais avec le module sklearn : {alpha_sk}")
        print(f"Poids de notre méthode : {w} et poids avec le module sklearn : {w_sk}")

#Nous remarquons qu'on a une légère différence entre les résultats des deux méthodes. 
#Nous pouvons expliquer cet écart minime par les limitations des ordinateurs pour calculer 
#avec des nombres à virgules. De plus le module sklearn est implenté de manière plus optimisé
#pour manipuler ce genre d'objets.


############
#Exercice 2

#Question 1 a
def generationW(n) :
    W = np.matrix(np.random.rand(n, 1))
    return W

#Question 1 b
def generationX(N,n,M):
    X = np.matrix(np.random.uniform(-M,M,(N,n)))
    return X

#A titre indicatif pour vérifier si mes fonctions marchent
#nous posons W,X et y et nous les utiliserons pour la suite (1)

W = generationW(3)
X = generationX(4,3,5)

#Question 1 c
y = X * W
print(f"y = {y}")

#Question 2 a Preuve : feuille manuscrite

#Question 2 a Python :
def F_exo2(w):
    f = (np.linalg.norm(X*w-y))**2
    return f

##Question 2 b Preuve : feuille manuscrite

#Question 2 ccPython :
A = X.T * X
z = X.T * y

#Question 2 d Calcul : feuille manuscrite

#Question 2 d Python :
W_opti = np.linalg.solve(A,z)

#On remarque qu'on trouve bien les poids W.

############
#Exercice 3
s = 0.01

#Question 1
def generation_y(M, s, W, X):
    y = X * W
    N,n = X.shape
    bruit = np.matrix(np.random.uniform(-s*M,s*M, (N,1)))
    y_bruite = y + bruit
    return y_bruite

#Nous posons notre nouveau vecteur y avec les matrices A et z en utilisant les anciennes données:

y_bruite = generation_y(5,s,W,X)
A = X.T * X
z_bruite = X.T * y_bruite

#Question 2
def F_exo3(w):
    f = (np.linalg.norm(X*w-y_bruite))**2
    return f

W_opti_bruite = np.linalg.solve(A,z_bruite)

#Question 3 a
s_exo3_a = 0.10
y_bruite_10 = generation_y(5,s_exo3_a,W,X)
z_bruite_10 = z_bruite = X.T * y_bruite_10

W_opti_bruite_10 = np.linalg.solve(A,z_bruite_10)

#Question 3 b
s_exo3_b = 0.50
y_bruite_50 = generation_y(5,s_exo3_b,W,X)
z_bruite_50 = z_bruite = X.T * y_bruite_50

W_opti_bruite_50 = np.linalg.solve(A,z_bruite_50)

############
#Exercice 4

#Question 1
nbIn = 3
nbOut= 1
model=torch.nn.Linear(nbIn,nbOut)

#Question 2
print(f"Le poids du réseau : {model.weight}") 
print(f"Le biais du réseau : {model.bias}") 

#Question 3 a
W_exo4 = np.random.random((3, 1))
X_exo4 = np.random.random((3, 3))
alpha_exo4 = 0
sigma = 0

def generationy_exo4(X_exo4, sigma,alpha_exo4,W_exo4):
    y = X_exo4 @ W_exo4
    return y
y_exo4 = generationy_exo4(X_exo4, sigma,alpha_exo4,W_exo4)

#Question 3 b
input = torch.FloatTensor(X_exo4)
target= torch.FloatTensor(y_exo4)

#Question 4 a
output = model(input)

#Question 4 b
w_convert_matrice = np.matrix((model.weight.detach().numpy()).T)
alpha_convert_float = float(model.bias.detach().numpy().item())

#Question 4 c
output_verif = X_exo4*w_convert_matrice + alpha_convert_float
print(output_verif)

#On remarque qu'on a bien output_verif = output

#Question 5
def Cout_F_exo4(output, target):
    return torch.nn.MSELoss()(output, target).item()

#Question 6
opti = torch.optim.SGD(model.parameters(), lr=0.01)

#Question 7
iterations = 1000
for i in range(iterations):
    opti.zero_grad()
    input = input = torch.FloatTensor(X_exo4)
    output = model(input)
    target = torch.FloatTensor(y_exo4)
    loss = Cout_F_exo4(output, target)
    print(loss)
    loss_forme_tensor = torch.nn.MSELoss()(output, target)
    loss_forme_tensor.backward()
    opti.step()

#Question 8
def verification_poinds_et_biais(model, W_exo4):
    poids_optimises = model.weight.detach().numpy().T
    biais_model = model.bias.detach().numpy().item()
   
    if np.allclose(poids_optimises, W_exo4.T, atol=1e-10) and abs(biais_model) < 1e-10:
        print("Sont tres proches")
        print(f"Poids reels : {W_exo4.T}")
        print(f"Poids optimises : {poids_optimises}")
        print(f"Biais : {biais_model}")
    else:
        print("Difference legere mais l'assertion reste vraie")
        print(f"Poids reels : {W_exo4.T}")
        print(f"Poids optimises : {poids_optimises}")
        print(f"Biais : {biais_model}")

verification_poinds_et_biais(model, W_exo4)

#Question 9 a
cout=[]
iterations = 1000
for i in range(iterations):
    opti.zero_grad()
    output = model(input)
    target = target
    loss = Cout_F_exo4(output, target)
    cout.append(loss)
    #print(loss)
    loss_tensor = torch.nn.MSELoss()(output, target)
    loss_tensor.backward()
    opti.step()

#Question 9 b
plt.plot(cout)
plt.xlabel("Iter")
plt.ylabel("Coût")
plt.title("Évolution du coût")
plt.show()

#Question 10
#En changeant les paramètres dans nos fonctions, nous remarquons qu'augmenter le taux d'apprentissage accélère l'apprentissage
# mais nous constatons une "divergence" des valeurs du poids du modèle, il n'y a pas de convergence vers une valeur optimale. 
# En revanche l'effets d'une diminutions du taux d'apprentissage est le ralentissement de la convergence des poids vers les valeurs
# optimales, de ce fait nous avons une convergence des poids mais lente ce qui peut être contraignant.



