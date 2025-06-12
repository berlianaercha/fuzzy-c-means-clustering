import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from factor_analyzer.factor_analyzer import calculate_kmo
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Fungsi validasi: Xie-Beni Index
def xie_beni_index(data, centers, u, m=2):
    n_clusters = centers.shape[0]
    N = data.shape[0]
    dist = euclidean_distances(data, centers) ** 2
    numerator = np.sum((u.T ** m) * dist)
    min_dist = np.min([
        np.linalg.norm(centers[i] - centers[j]) ** 2
        for i in range(n_clusters) for j in range(n_clusters) if i != j
    ])
    xb_index = numerator / (N * min_dist)
    return xb_index

# Buat dua tab
tab1, tab2 = st.tabs(["Informasi Penyusun dan Aplikasi", "Fuzzy C-Means Clustering"])

# ====================== TAB 1 ======================
with tab1:
    st.title("Informasi Penyusun dan Aplikasi")

    st.markdown("""
    #### Penyusun:
    - Nama: Berliana Ercha Pratiwi
    - NIM: 24050122120011
    - Kelas: Komputasi Statistika Lanjut - D

    ### Aplikasi Fuzzy C-Means Clustering
    Aplikasi ini digunakan untuk melakukan clustering data menggunakan algoritma Fuzzy C-Means (FCM).

    #### Fitur Utama:
    - Upload dan preview dataset (.csv)
    - Uji kelayakan data menggunakan KMO Test (untuk data sampel)
    - Deteksi multikolinearitas menggunakan VIF
    - Clustering menggunakan Fuzzy C-Means
    - Validasi hasil clustering dengan Xie-Beni Index
    - Visualisasi hasil clustering 
    - Profilisasi hasil clustering

    Aplikasi ini dikembangkan menggunakan Python dan Streamlit.
    """)

# ====================== TAB 2 ======================
with tab2:
    st.title("Fuzzy C-Means Clustering")

    uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview")
        df.index = np.arange(1, len(df) + 1)
        st.dataframe(df)

        with st.expander("Step 1: Pilih Kolom Nama Objek"):
            object_column = st.selectbox(
                "Pilih kolom yang berisi nama/ID objek (tidak digunakan dalam clustering):",
                options=df.columns
            )

        with st.expander("Step 2: Tentukan Jenis Data"):
            data_type = st.radio("Apakah data berupa sampel atau populasi?", ["Sampel", "Populasi"])

        if data_type == "Sampel":
            st.subheader("Step 3: Uji KMO")
            try:
                data_kmo = df.drop(columns=[object_column], errors='ignore').dropna()
                kmo_all, kmo_model = calculate_kmo(data_kmo)
                st.write(f"KMO: {kmo_model:.4f}")
                if kmo_model >= 0.5:
                    st.success("KMO mencukupi untuk analisis lebih lanjut.")
                else:
                    st.warning("KMO < 0.5: Data tidak cocok untuk clustering.")
            except Exception as e:
                st.error(f"Uji KMO gagal: {e}")
                st.stop()

        st.subheader("Step 4: Uji Non-Multikolinearitas (VIF)")
        try:
            vif_data = df.drop(columns=[object_column], errors='ignore').dropna()
            scaler = StandardScaler()
            vif_scaled = scaler.fit_transform(vif_data)

            vif_df = pd.DataFrame()
            vif_df["Variable"] = vif_data.columns
            vif_df["VIF"] = [variance_inflation_factor(vif_scaled, i) for i in range(vif_scaled.shape[1])]

            st.write("Nilai VIF (Variance Inflation Factor):")
            vif_df.index = np.arange(1, len(vif_df) + 1)
            st.dataframe(vif_df)

            if vif_df["VIF"].max() > 10:
                st.warning("Terdapat variabel dengan VIF > 10: kemungkinan multikolinearitas tinggi.")
            else:
                st.success("Semua variabel memiliki VIF < 10: tidak ada indikasi multikolinearitas tinggi.")

        except Exception as e:
            st.error(f"Uji VIF gagal: {e}")
            st.stop()

        st.subheader("Step 5: Fuzzy C-Means Clustering")
        num_clusters = st.slider("Jumlah Cluster (c)", min_value=2, max_value=10, value=3)
        m = st.slider("Tingkat Fuzziness (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
        max_iter = st.slider("Maksimum Iterasi", min_value=100, max_value=1000, value=300, step=50)

        if st.button("Lakukan Clustering"):
            try:
                data_all = df.dropna()
                objek_ids = data_all[object_column].values if object_column in data_all.columns else None

                data = data_all.drop(columns=[object_column], errors='ignore')
                data_raw = data.values
                data_T = data_raw.T

                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data_T, c=num_clusters, m=m, error=10e-09, maxiter=max_iter, init=None
                )

                cluster_labels = np.argmax(u, axis=0)
                df_result = data_all.copy()
                df_result["Cluster"] = cluster_labels + 1

                for i in range(num_clusters):
                    df_result[f'Membership_in_Cluster_{i + 1}'] = u[i]

                for col in ["SD", "SMP", "SMA"]:
                    if col in df_result.columns:
                        df_result.drop(columns=col, inplace=True)

                dist_matrix = euclidean_distances(data_raw, cntr)
                for i in range(num_clusters):
                    df_result[f'Distance_to_Center_{i + 1}'] = dist_matrix[:, i]

                st.success("Clustering Selesai!")
                st.write("Hasil Clustering:")
                df_result.index = np.arange(1, len(df_result) + 1)
                st.dataframe(df_result)

                st.subheader("Step 6: Validasi Clustering (Xie-Beni Index)")
                xb = xie_beni_index(data_raw, cntr, u, m)
                st.write(f"Xie-Beni Index: {xb:.4f}")
                st.caption("Semakin kecil nilai Xie-Beni, semakin baik pemisahan cluster.")

                st.subheader("Visualisasi Hasil Clustering")
                show_labels = False

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data_raw)
                df_result["PCA1"] = pca_result[:, 0]
                df_result["PCA2"] = pca_result[:, 1]

                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    df_result["PCA1"],
                    df_result["PCA2"],
                    c=df_result["Cluster"],
                    cmap='viridis',
                    s=100,
                    alpha=0.8,
                    edgecolors='k'
                )

                if show_labels and objek_ids is not None:
                    for i, obj_id in enumerate(objek_ids):
                        ax.text(df_result["PCA1"].iloc[i], df_result["PCA2"].iloc[i], str(obj_id),
                                fontsize=8, ha='right', va='bottom')

                ax.set_title("Visualisasi Hasil Clustering")
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")

                handles, _ = scatter.legend_elements()
                labels = [f"Cluster {i+1}" for i in range(num_clusters)]
                ax.legend(handles, labels, title="Cluster", loc="best")

                st.pyplot(fig)

                # ====================== Tambahan: Tabel Anggota Cluster ======================
                st.subheader("Tabel Ringkasan Anggota Setiap Cluster")

                cluster_groups = df_result.groupby("Cluster")[object_column].apply(list).reset_index()
                cluster_groups.columns = ["Cluster", "Anggota Cluster"]
                cluster_groups["Jumlah Anggota"] = cluster_groups["Anggota Cluster"].apply(len)
                cluster_groups.index = np.arange(1, len(cluster_groups) + 1)
                st.dataframe(cluster_groups)

                st.subheader("Tabel Detail: Setiap Objek dan Cluster-nya")
                long_cluster_table = df_result[[object_column, "Cluster"]].sort_values("Cluster").reset_index(drop=True)
                long_cluster_table.index = np.arange(1, len(long_cluster_table) + 1)
                st.dataframe(long_cluster_table)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat clustering: {e}")
