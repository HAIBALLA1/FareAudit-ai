
# Guide expérimental d'investigation FareAudit AI

Nature : règles exploratoires internes au projet.
Ces règles ne sont pas des règles officielles NYC TLC.

## Montants négatifs

Un total_amount négatif doit être examiné.

Il peut correspondre à une correction, une opération
particulière ou un problème de données.

Un montant négatif ne constitue pas à lui seul
une preuve de fraude.

## Distance nulle

Une course avec trip_distance = 0 peut être
signalée pour vérification.

Une distance nulle ne prouve pas nécessairement
une transaction incorrecte.

## Montant élevé

Dans notre prototype, total_amount > 200 dollars
est un indicateur exploratoire.

Le seuil de 200 dollars est arbitraire.
Une course longue ou particulière peut être légitime.

## Modèle Isolation Forest

Isolation Forest détecte des observations
statistiquement inhabituelles.

Un score decision_function négatif signifie
que le modèle classe le trajet comme anomalie.

Ce score n'est pas une probabilité de fraude.

## Méthode d'investigation

Pour examiner une transaction :
1. Identifier les caractéristiques inhabituelles.
2. Examiner le type de paiement.
3. Vérifier les champs disponibles.
4. Consulter les documents pertinents.
5. Présenter des hypothèses sans conclure à la fraude.
6. Indiquer les informations supplémentaires nécessaires.
