import matplotlib.pyplot as plt
import numpy as np
import random
import pulp
import sys
import time

#Utilitaires
def distancemanhattan(x1,y1,x2,y2):
    return abs(x1-x2)+abs(y1-y2)

def genererProbleme (n) :
    posClient=[0]*n 
    for i in range(n): 
        posClient[i] = [random.randint(0,n*n), random.randint(0,n*n),i+1]
    
    posDepot = [n*n//2, n*n//2, 0]
    posClient.insert(0,posDepot)
    return posClient

def solutionValide(n, posClient, posDepot, solution) : 
    if len(solution) != n+1 : return False
    if solution[0] != 0 : return False
    if solution[n] != 0 : return False
    for x in set(solution+posClient) : 
        if solution.count(x)!=posClient.count(x) : return False 
    return True
        

def matriceDistances(u) : 
    n = len(u)
    matrice = [[0]*n for i in range(n)]
    for i in range(n) : 
        for j in range(n) : 
            matrice[i][j] = distancemanhattan(u[i][0],u[i][1],u[j][0],u[j][1])
    return matrice

def sousEnsembles (e) : 
    if len(e) == 0 : return [[]]
    else : 
        sous = sousEnsembles(e[1:])
        return sous + [[e[0]]+x for x in sous]

def trifusionPile(x): #version sans récursion pour trifusion (j'ai regardé sur internet)
    if len(x) < 2:
        return x
    result = []
    y = trifusionPile(x[:len(x)//2])
    z = trifusionPile(x[len(x)//2:])
    i = 0
    j = 0
    while i < len(y) and j < len(z):
        if len(y[i]) > len(z[j]):
            result.append(z[j])
            j += 1
        else:
            result.append(y[i])
            i += 1
    result += y[i:]
    result += z[j:]
    return result


def trifusionsubsets(T) :#Trie l'ensemble des sous ensemble par taille
    if len(T)<=1 : return T
  
    T1=[T[x] for x in range(len(T)//2)]
    T2=[T[x] for x in range(len(T)//2,len(T))]
    return fusion2(trifusionsubsets(T1),trifusionsubsets(T2))
def fusion2(T1,T2) :
    if T1==[] :return T2
    if T2==[] :return T1

    if len(T1[0])<len(T2[0]) :
        return [T1[0]]+fusion2(T1[1 :],T2)
    else :
        return [T2[0]]+fusion2(T1,T2[1 :])

##



##FONCTION PRINCIPALE

def programmation_dynamique (probleme) : 
    n = len(probleme)
    distances = matriceDistances(probleme)
   
    s = {}
    subSizeCounter = n
    subsets = trifusionPile(sousEnsembles([x for x in range(1,n)]))#On calcule tous les sous ensembles de [1..n] et on les trie par taille croissante
    
    #dans S : (representation binaire du sous ensemble, noeud) -> (distance, noeud precedent) 
    for i in range(1,n) :
        #Sous ensembles sont ecrits en binaire : [1,5] = 100010, [3] = 1000
        s[(1<<i,i)]= (distances[0][i],0)
        
    for taille in range(2,n) : #On itere sur la taille des sous ensembles
        while subSizeCounter != len(subsets) and len(subsets[subSizeCounter])==taille : #le while evite 
            #de faire un for et permet de ne pas parcourir tous les sous ensembles lors de la recherche des sous ensembles de taille taille (car la liste des sous ensembles est triée)
           
            representationBin = 0 
            for noeud in subsets[subSizeCounter] :#On represente le sous ensemble total de taille taille
                representationBin += 1<<noeud 
                
            for j in subsets[subSizeCounter] :
                exclu = representationBin - (1<<j)  #on enleve le noeud j du sous ensemble 
                listemin = []
                
                for k in subsets[subSizeCounter] : #on stocke toutes les distances mises à jour des différents sous ensembles 
                    if j!=k and k!=0 :
                        listemin.append((s[(exclu,k)][0]+distances[k][j],k))
                s[(representationBin,j)] = min(listemin)
            subSizeCounter +=1
        
        
    
    total = 2**n - 2 #On prend les sous ensembles qui sont complets 
    listemin2 = []
    for i in range(1,n) : #On ajoute tous les derniers couts (mis à jour au passage avec la derniere distance) pour calculer leur minimum
        listemin2.append((s[(total,i)][0]+distances[i][0], i))
    
    (minoptimal, parent) = min(listemin2)
    
    chemin = [] #On retrouve le chemin emprunté 
    for i in range(n-1) : 
        chemin.append(parent)
        repres = total - (1<<parent)
        (a,parent) = s[(total,parent)]
        total = repres
    
    chemin.append(0)
    return minoptimal, chemin[::-1] #on inverse le chemin pour avoir le bon sens 



####
def genererCheminAleatoire(listeSommets) : 
    chemin = [listeSommets[0]]
    while len(chemin) != len(listeSommets) : 
        chemin.append(random.choice([x for x in listeSommets if x not in chemin]))
    chemin.append(listeSommets[0])
    return chemin

def calculCout(chemin): 
    cout = 0
    for i in range(len(chemin)-1) : 
        cout += distancemanhattan(chemin[i][0],chemin[i][1],chemin[i+1][0],chemin[i+1][1])
    return cout

def coupe_relie(chemin, i, j) : 
    chemin2 = []
    if i>j : 
        i,j = j,i
    chemin2 = chemin[:i+1] + [chemin[j]] + chemin[j-1:i:-1] + chemin[j+1:]
    
    return chemin2

#On peut mettre un chemin prédéfini (utile si on veut améliorer un parcours préfixe par exemple)
def recherchelocale(probleme, chemin = None, temps = 3) : 
    st = time.time()
    distances = matriceDistances(probleme)
   

    if chemin == None : 
        chemin = genererCheminAleatoire(probleme)
    i = 1 
    j = 3
    chemins = []

    while(time.time()-st<temps):
        cout = calculCout(chemin)
        i=1
        j=3
        while i<len(chemin)-2 and j!=len(chemin)-1:
            chemin2 = coupe_relie(chemin,i,j)
            #maj du cout a l'aide de la formule sur le poly. 
            cout2 = cout - distances[chemin[i][2]][chemin[i+1][2]] - distances[chemin[j][2]][chemin[j+1][2]] + distances[chemin[i][2]][chemin[j][2]] + distances[chemin[i+1][2]][chemin[j+1][2]]
            if cout2 < cout : 
                cout = cout2
                chemin = chemin2
                i = 1
                j = 3
            elif j+1<len(chemin)-1 :
                j+=1
            else :
                i+=1
                j = i+2
        
        for i in range(len(chemin)) : 
            chemin[i] = chemin[i][2:]
        chemins.append([cout,chemin])
        chemin = genererCheminAleatoire(probleme)
        
        cout = 0
    
    return min(chemins)
            
   


###########




#Definition du probleme comme indiqué dans le poly. 
def optimisationLineaire(pb) :
    distances = matriceDistances(pb) 
    probleme = pulp.LpProblem("Probleme_du_voyageur_de_commerce",pulp.LpMinimize)
    n = len(pb)
    x = pulp.LpVariable.dicts("x",((i,j) for i in range(n) for j in range(n) if i !=j),0,1,pulp.LpInteger)
    u = pulp.LpVariable.dicts("u", (i for i in range(n)),None,None,pulp.LpInteger)
    probleme += pulp.lpSum([x[i,j]*distances[i][j] for i in range(n) for j in range(n) if i!=j])
    
    for i in range(1,n) : 
        probleme += pulp.lpSum([x[i,j] for j in range(n) if j!=i]) == 1
    for j in range(1,n) :
        probleme += pulp.lpSum([x[i,j] for i in range(n) if j!=i]) == 1
    for i in range(1,n) :
        for j in range(1,n) :
            if i!=j  : 
                probleme += u[i]-u[j]+(n-1)*x[i,j] <= n-2
    return probleme.solve()
    
######## Classe pour pouvoir appliquer kruskal facilement
class EnsembleKruskal : 
    parent = {}

    def creerEnsemble (self,n) :
        for i in range(n) : 
            self.parent[i] = i
    def getParent(self,k) : 
        
        while self.parent[k] != k : 
            k = self.parent[k]
        return k
        
    def union(self,i,j) : 
        x = self.getParent(i)
        y = self.getParent(j)
        self.parent[x] = y
##########
def constructionEnsembleArcs(pb) : 
    arcs = []
    for i in range(len(pb)) : 
        for j in range(len(pb)) : 
            if i!=j : 
                arcs.append((i,j,distancemanhattan(pb[i][0],pb[i][1],pb[j][0],pb[j][1])))
    
    return arcs

def kruskal(pb) : 
    arcs = constructionEnsembleArcs(pb)
    arbre = []
    ensemble = EnsembleKruskal();
    ensemble.creerEnsemble(len(pb))
    i = 0 
    arcs.sort(key = lambda x : x[2])#on trie par ordre de poids croissant
    while len(arbre) != len(pb)-1 :
        (debut,fin,poids) = arcs[i]
        i +=1
      
        parentsource = ensemble.getParent(debut)
      
    
        parentfin = ensemble.getParent(fin)
       
        if parentsource != parentfin : 
            arbre.append((debut,fin, poids))
            ensemble.union(parentsource,parentfin)
    return arbre


#parcours prefixe de l'arbe... surement pas optimal, j'ai evité les recursions
def parcoursPrefixe(arbre) : 
    if len(arbre) == 0 :
        return []
    copiearbre=[]
    for i in range(len(arbre)) : 
        copiearbre.append(arbre[i])
    parcours = []
    stack =[]
    stack.append(copiearbre[0])
    (debut,fin,poids) = copiearbre[0]
    stack.append(debut)
    stack.append(fin)
    parcours.append(debut)
    parcours.append(fin)
    removelist = [copiearbre[0]] #Liste des sommets à enlever. On les enleve à chaque itération du while et pas pendant car on en a besoin tout au long de l'execution 
                                 

    while(len(stack)!=0): 
        copiearbre = [x for x in copiearbre if x not in removelist]
        debut = stack.pop()
       
        for i in range(len(copiearbre))  : 
            #On regarde si le début ou la fin (car le graphe n'est pas orienté) d'un des arcs du sommet courant va vers un autre sommet de l'arbre. 
            if copiearbre[i][0] == debut : 
                stack.append(copiearbre[i][1])
                parcours.append(copiearbre[i][1])
                removelist.append(copiearbre[i])
                
            elif copiearbre[i][1] == debut : 
                stack.append(copiearbre[i][0])
                removelist.append(copiearbre[i])
                parcours.append(copiearbre[i][0])
    for i in range(len(parcours)) :
        if parcours[i] == 0 : 
            parcoursReturn = parcours[i:]+parcours[:i]
            break
    
    return parcoursReturn

def approximation(pb) : 
    distances = matriceDistances(pb)
    cout = 0 
    arbre = kruskal(pb)
    parcours = parcoursPrefixe(arbre)
    for i in range(len(parcours)) : 
        if i != len(parcours)-1 : 
            cout +=distances[parcours[i]][parcours[i+1]]
            
    return (cout,parcours)



"""
On découpe notre carré de taille 1000000*1000000 en 10 parties définies par les droites coupant le centre du carré de telle maniere à ce que chaque droite définisse avec
sa voisine un angle entre 30 et 44 degrés pour diminuer les differences entre les zones contenant des coins.
 De cette manière le carré est découpé en 10 parties plus ou moins égales, mais d'ordres de grandeur équivalents.
L'important est que chaque partie est presque aussi profonde qu'une autre (sauf pour les coins..). Plus de détails dans le
rapport.

On suppose que le carré est centré en (500000,500000) et donc que les coordonnées des points sont (1000000, 1000000), (0,1000000), (1000000, 0), (0,0).
Equations de droites : 
triangle superieur gauche : -0.78x + +890642.81 < y < -2.48x + 1737543.43
triangle milieu superieur :   -2.48x + 1737543.43 < y et 2,48x - 737543.43 < y
triangle superieur droit :     0.78x + +109357.19 < y<  2,48x - 737543.43
triangle droit dans la partie superieure du carré: 500000 <y< 0.78x + +109357.19
triangle droit dans la partie inférieure du carré :  -0.78x + +890642.81 <y < 500000
triangle inferieur droit :  -2.48x + 1737543.43 <y<-0.78x + +890642.81 
triangle milieu inferieur : y < -2.48x + 1737543.43  et y<  2,48x - 737543.43 
triangle inferieur gauche :    2,48x - 737543.43 < y < 0.78x + +109357.19 
triangle gauche dans la partie inferieure du carré :  0.78x + +109357.19 < y < 500000
triangle gauche dans la partie superieure du carré : 500000<y< -0.78x + +890642.81

"""



def repartirPoints(pb) :
    supGauche = [[500000,500000,0]]
    milieuSup =[[500000,500000,0]]
    supDroit=[[500000,500000,0]]
    droitSupCarre = [[500000,500000,0]]
    droitInfCarre = [[500000,500000,0]]
    infDroit =[[500000,500000,0]]
    milieuInf = [[500000,500000,0]]
    infGauche = [[500000,500000,0]]
    gaucheInfCarre = [[500000,500000,0]]
    gaucheSupCarre =[[500000,500000,0]]
    for i in range(1,len(pb)) : 

        #On regarde dans quelles zones du carré sont les points. 
        if pb[i][0]*-0.78 + 890642.81 < pb[i][1] and -2.48*pb[i][0] + 1737543.43 > pb[i][1] :
            supGauche.append(pb[i])

        elif pb[i][0]* -2.48 + 1737543.43 < pb[i][1] and 2.48*pb[i][0] - 737543.43 < pb[i][1] :
            milieuSup.append(pb[i])
            
        elif pb[i][1] > 500000 and pb[i][0]*0.78 + 109357.19 > pb[i][1] :
            droitSupCarre.append(pb[i])
        
        elif pb[i][0]*-0.78 + 890642.81 < pb[i][1] and pb[i][1] < 500000 :
            droitInfCarre.append(pb[i])
           
        elif pb[i][0]*-2.48 + 1737543.43 < pb[i][1] and pb[i][0]*-0.78 + 890642.81 > pb[i][1] :
            infDroit.append(pb[i])
           
        elif pb[i][0]*2.48 - 737543.43 > pb[i][1] and pb[i][0]*-2.48 + 1737543.43 > pb[i][1] :
            milieuInf.append(pb[i])
            
        elif pb[i][0]*2.48 - 737543.43 < pb[i][1] and pb[i][0]*0.78 + 109357.19 > pb[i][1] :
            infGauche.append(pb[i])
         
        elif pb[i][1] < 500000 and pb[i][0]*0.78 + 109357.19 < pb[i][1] :
            gaucheInfCarre.append(pb[i])
           
        elif pb[i][1] > 500000 and pb[i][0]*-0.78 + 890642.81 > pb[i][1] :
            gaucheSupCarre.append(pb[i])

        elif pb[1][0]*0.78 + 109357.19 < pb[i][1] and 2.48*pb[i][0] - 737543.43 > pb[i][1] :
            supDroit.append(pb[i])
    #print (len(supGauche),len(milieuSup),len(supDroit),len(droitSupCarre),len(droitInfCarre),len(infDroit),len(milieuInf),len(infGauche),len(gaucheInfCarre),len(gaucheSupCarre))
    return [supGauche,milieuSup,supDroit,droitSupCarre,droitInfCarre,infDroit,milieuInf,infGauche,gaucheInfCarre,gaucheSupCarre]

def pbAvec10CoursiersApproximation(pb, temps = 0.1) : 
    problemes = repartirPoints(pb)
    problemesBonsIndices = [0]*len(problemes)
    for i in range(len(problemes)):
            problemesBonsIndices[i]=problemes[i]
    #On met des indices de 0 à la taille du sous ensemble pour que les algorithmes s'effectuent correctement.
    for i in range(len(problemesBonsIndices)):
        for j in range(len(problemesBonsIndices[i])):
            [a,b,c] = problemesBonsIndices[i][j]
            problemesBonsIndices[i][j] = [a,b,j,c]       
    solutions =[]
    #On effectue l'algorithme d'approximation sur chaque sous ensemble en faisant un parcours prefixe. 
    for i in range(len(problemesBonsIndices)) :
        parcours= parcoursPrefixe(kruskal(problemesBonsIndices[i]))
        parcoursAvecCoordonnees =[]*len(parcours)
       
        for k in range(len(parcours)) : 
            parcoursAvecCoordonnees.append(problemesBonsIndices[i][parcours[k]])
        #On effectue la recherche locale sur chaque parcours prefixe fourni pour améliorer les distances si possible. 
        solutions.append(recherchelocale(problemesBonsIndices[i],parcoursAvecCoordonnees ,temps =temps))    
  
    #recuperation et suppression des indices artificiels. 
    for i in range(len(solutions)):
        for j in range(1,len(solutions[i])):
            for k in range(len(solutions[i][j])):
                solutions[i][j][k]=solutions[i][j][k][1]
        
        
    return (solutions)

#meme fonction que approximation mais avec uniquement la recherche locale.
def pbAvec10CoursiersRechercheLocale(pb, temps) :
    problemes = repartirPoints(pb)
    problemesBonsIndices = [0]*len(problemes)
    for i in range(len(problemes)):
        problemesBonsIndices[i]=problemes[i]
    for i in range(len(problemesBonsIndices)):
        for j in range(len(problemesBonsIndices[i])):
            [a,b,c] = problemesBonsIndices[i][j]
            problemesBonsIndices[i][j] = [a,b,j,c]       
    solutions =[]
    for i in range(len(problemesBonsIndices)) : 
        solutions.append(recherchelocale(problemesBonsIndices[i],temps =temps))    
    for i in range(len(solutions)):
        for j in range(1,len(solutions[i])):
            for k in range(len(solutions[i][j])):
                solutions[i][j][k]=solutions[i][j][k][1]

    return solutions

def comparaisonsLocaleApprox(iterations, taille, temps = 0.05) : 
    print("Comparaisons pour ", iterations, "iterations et problemes de taille", taille)
    recherche = 0 
    ecartR = 0

    approx = 0
    ecartA = 0 
    egal = 0 
    for i in range (iterations) : 
        a = genererProbleme(taille)
    
        (c, c2) = recherchelocale(a, temps = temps)
        (c3,c4) = approximation(a)
        if c<c3 : 
            recherche +=1
            ecartR += c3-c
        if c3<c : 
            approx +=1
            ecartA+=c-c3
        if c == c3 : 
            egal +=1
    ecartR =ecartR/recherche
    if approx!=0 : 
        ecartA =ecartA/approx
    print("la recherche locale est meilleure pour 0.05 secondes", recherche, "fois")
    print("l'approximation  est inférieure ", approx, "fois")
    print("egalité ", egal, "fois")
    print("ecart moyen quand la recherche locale gagne : ", ecartR)
    print("ecart moyen quand l'approximation gagne : ", ecartA)


