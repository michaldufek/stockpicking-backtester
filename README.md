# ETL Process: ./src/datafeed.sh
- set your path to the script 
```
cd ~/YOUR/PATH/stockpicking-model/src/
```
- set cron job in your system
```
crontab -e
```
```
# m h  dom mon dow   command
15 22 * * * /YOUR/PATH/stockpicking-model/src/datafeed.sh
```

- do not forget check your timezone
```
cat /etc/timezone 
Europe/Prague
```