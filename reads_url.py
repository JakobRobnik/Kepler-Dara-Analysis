from urllib.request import urlopen
url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=q1_q17_dr24_tce&select=rowid,kepid,tce_plnt_num,tce_period,tce_time0bk,tce_duration,av_training_set&format=csv'
response = urlopen(url)
html = response.read()

with open('names.csv', 'wb') as f:
    f.write(html)