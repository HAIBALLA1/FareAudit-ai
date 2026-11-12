
# Types de paiement des taxis jaunes NYC

Source officielle :
https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

## Codes des types de paiement

La colonne payment_type décrit le mode ou le statut de paiement d'un trajet.

0 = Flex Fare trip (tarification flexible)
1 = Credit card (carte bancaire)
2 = Cash (espèces)
3 = No charge (sans facturation)
4 = Dispute (litige)
5 = Unknown (inconnu)
6 = Voided trip (trajet annulé)

## Interprétation

Le code payment_type = 4 correspond à un litige.

Le code payment_type = 3 correspond à une course sans facturation.

Ces codes décrivent la transaction. Ils ne prouvent pas
à eux seuls qu'une fraude a eu lieu.
