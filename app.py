import pandas as pd
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Baca dataset
df = pd.read_csv('dataset.csv')  # Ganti 'dataset.csv' dengan nama file dataset Anda

# Route untuk halaman beranda
@app.route('/')
def beranda():
    return render_template('beranda.html')

# Route untuk halaman sentimen
@app.route('/sentimen')
def sentimen():
    # Menghitung jumlah data dengan label positif, negatif, dan netral
    positif_count = len(df[df['Label'] == 'Positif'])
    negatif_count = len(df[df['Label'] == 'Negatif'])
    netral_count = len(df[df['Label'] == 'Netral'])
    
    # Menghitung total data dalam dataset
    total_data = len(df)
    
    # Menghitung persentase
    positif_percentage = (positif_count / total_data) * 100
    negatif_percentage = (negatif_count / total_data) * 100
    netral_percentage = (netral_count / total_data) * 100
    
    return render_template('sentimen.html', 
                           positif=positif_percentage, negatif=negatif_percentage, netral=netral_percentage)

from flask import jsonify

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        filtered_df = df[df['Teks'].str.contains(query, case=False, na=False)]

        # Menghitung jumlah data dengan label positif, negatif, dan netral
        positif_count = len(filtered_df[filtered_df['Label'] == 'Positif'])
        negatif_count = len(filtered_df[filtered_df['Label'] == 'Negatif'])
        netral_count = len(filtered_df[filtered_df['Label'] == 'Netral'])

        # Menghitung persentase sentimen
        total_sentimen = positif_count + negatif_count + netral_count
        positif_percentage = (positif_count / total_sentimen) * 100
        negatif_percentage = (negatif_count / total_sentimen) * 100
        netral_percentage = (netral_count / total_sentimen) * 100

        # Mengambil hasil pencarian dalam bentuk dictionary
        search_results = filtered_df.to_dict(orient='records')

        return jsonify({
            'search_results': search_results,
            'positif': positif_percentage,
            'negatif': negatif_percentage,
            'netral': netral_percentage,
            'query': query
        })
    else:
        return jsonify({
            'error': 'Kata kunci pencarian tidak ditemukan'
        })


if __name__ == '__main__':
    app.run(debug=True)
