#!/bin/bash
##################################################
# !!!DEFINE YOUR PATH TO THIS SCRIPT!!!
##################################################
if cd ~/Documents/stockpicking-model/src/
then
    :
else
echo "YOU HAVE TO SPECIFY PATH IN DATAFEED.SH SCRIPT" && exit 1
fi
##################################################
# PRICE DATA PUMP
##################################################

# FRS DEV VPS & Entry to docker and copy current data
ssh root@206.189.58.244 "docker cp frs-django-dev:/usr/src/app/data/dwh_data/ ~/workspace/research_data/"

# Copy data (folder with files) from FRS DEV VPS to local machine
scp -r root@206.189.58.244:~/workspace/research_data/dwh_data/ ../SP-data/
# Remove "obsolete" data if does exist
if [[ -d "../SP-data/price_data" ]]
then
    echo "Removing obsolete price data"
    rm -r ../SP-data/price_data/
fi
# Rename folder structure and gain the new data
mv ../SP-data/dwh_data/ ../SP-data/price_data/ && echo "Renaming folder structure for price ../SP-data/price_data"


# Remove ~/workspace/research_data/* files
#ssh root@206.189.58.244 "rm ~/workspace/research_data/* && rmdir ~/workspace/research_data"

##################################################
# FUNDAMENTAL DATA PUMP
##################################################
scp -r ib@46.101.242.92:/home/data/fundamental_data/firm_specific/daily_yahooFinance/ ../SP-data/fundamental_data/firm_specific/
# Remove "obsolete" data
if [[ -d "../SP-data/fundamental_data/firm_specific/feed_fundamental/" ]]
then
    echo "Removing obsolete fundamental data..."
    rm -r ../SP-data/fundamental_data/firm_specific/feed_fundamental/
fi
# Rename folder structure
mv ../SP-data/fundamental_data/firm_specific/daily_yahooFinance/ ../SP-data/fundamental_data/firm_specific/feed_fundamental/ && echo "Renaming folder structure for fundamental"

##################################################
# FUNDAMENTAL DATA PROCESSOR
##################################################
python3 fundamental_processor.py