import pandas as pd

df = pd.read_csv("C:/Users/USER/Documents/Physics/kepler/download_skripte/poimenovanja.csv")
idji_kandidatov = df.query('tce_plnt_num == "1"')['kepid'].tolist()
neustrezni_kandidati = df.query('tce_plnt_num  != "1" & kepid in @idji_kandidatov')['kepid'].tolist()
print(len(idji_kandidatov))
for neustrezen in neustrezni_kandidati:
    while neustrezen in idji_kandidatov:
        idji_kandidatov.remove(neustrezen)
print(len(idji_kandidatov))
arr = df[df['kepid'].isin(idji_kandidatov[200:600])]
arr.to_csv('ne_planeti600.csv', index=False)
print(idji_kandidatov[200:600])

# period = arr['tce_period'].tolist()
# phase = (arr['tce_time0bk']-131.512439459).tolist()
# time_half_transit = (arr['tce_period'] *0.5 /24.0).tolist()
