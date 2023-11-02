import pandas as pd
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Baca dataset
df = pd.read_csv('dataset.csv')  # Ganti 'dataset.csv' dengan nama file dataset Anda


@app.route('/')
def sentimen():
    # Menghitung jumlah data dengan label positif, negatif, dan netral
    positif_count = len(df[df['Label'] == 'Positif'])
    negatif_count = len(df[df['Label'] == 'Negatif'])
    netral_count = len(df[df['Label'] == 'Netral'])
    
    # Menghitung total data dalam dataset
    total_data = len(df)
    
    # Menghitung persentase
    positif_percentage = "{:.2f}%".format((positif_count / total_data) * 100)
    negatif_percentage = "{:.2f}%".format((negatif_count / total_data) * 100)
    netral_percentage = "{:.2f}%".format((netral_count / total_data) * 100)
    
    return render_template('sentimen.html', 
                           positif=positif_percentage, negatif=negatif_percentage, netral=netral_percentage)
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        filtered_df = df[df['Teks'].str.contains(query, case=False, na=False)]

        # Menghitung jumlah data dengan label positif, negatif, dan netral
        positif_count = len(filtered_df[filtered_df['Label'] == 'Positif'])
        negatif_count = len(filtered_df[filtered_df['Label'] == 'Negatif'])
        netral_count = len(filtered_df[filtered_df['Label'] == 'Netral'])

        # Menghitung total sentimen
        total_sentimen = positif_count + negatif_count + netral_count

        if total_sentimen == 0:
            positif_percentage = "0.00%"  # Atur nilai default ke nol jika total_sentimen adalah nol
            negatif_percentage = "0.00%"
            netral_percentage = "0.00%"
        else:
            # Menghitung persentase sentimen
            positif_percentage = "{:.2f}%".format((positif_count / total_sentimen) * 100)
            negatif_percentage = "{:.2f}%".format((negatif_count / total_sentimen) * 100)
            netral_percentage = "{:.2f}%".format((netral_count / total_sentimen) * 100)

        # Mengambil hasil pencarian dalam bentuk dictionary
        search_results = filtered_df.to_dict(orient='records')

        return render_template('sentimen.html', 
                           positif=positif_percentage, negatif=negatif_percentage, netral=netral_percentage, search_results=search_results)
    else:
        return jsonify({
            'error': 'Kata kunci pencarian tidak ditemukan'
        })

if __name__ == '__main__':
    app.run(debug=True)
