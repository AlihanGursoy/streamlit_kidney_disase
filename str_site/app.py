# app.py
# -*- coding: utf-8 -*-
import os, io, datetime
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ================== SABİTLER ==================
MODEL_PATH = "/final_pipeline_RandomForest.joblib"  # sabit model yolu
EXCEL_LOG_PATH = "/tahminler.xlsx"                                   # otomatik yazılacak Excel
COVER_IMAGE_CANDIDATES = ["/tas.jpg"]

# ========== Eğitimde kullanılan FunctionTransformer ==========
def to_str_array(X):
    import pandas as pd
    if isinstance(X, pd.Series):  return X.astype(str).to_frame()
    if isinstance(X, pd.DataFrame): return X.astype(str)
    return pd.DataFrame(X).astype(str).values
# ============================================================

st.set_page_config(page_title="Böbrek Taşı — ML Destek", layout="wide")

# ---- Session state ----
for k, v in {
    "page": "home",
    "pkg": None, "pipe": None, "classes": None, "feat_cols": None,
    "cover_bytes": None,
    "records": []   # bu oturumdaki tahminler (ekranda kalsın)
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------------- Yardımcılar ----------------
def find_cover_bytes():
    if st.session_state.cover_bytes:
        return st.session_state.cover_bytes, "uploaded"
    for p in COVER_IMAGE_CANDIDATES:
        if os.path.exists(p):
            with open(p, "rb") as f: return f.read(), p
    return None, None

def ensure_model_loaded(reload: bool = False):
    if (st.session_state.pipe is None) or reload:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model dosyası bulunamadı: **{MODEL_PATH}**"); st.stop()
        try:
            pkg = load(MODEL_PATH)
            st.session_state.pkg = pkg
            st.session_state.pipe = pkg["pipeline"]
            st.session_state.classes = pkg["label_classes"]
            st.session_state.feat_cols = list(pkg["feature_columns"])
        except Exception as e:
            st.error(f"Model yüklenemedi: {e}"); st.stop()

def code_from_label(s: str) -> int:
    import re
    m = re.search(r"\((\-?\d+)\)", s)
    if m: return int(m.group(1))
    try: return int(s)
    except: return 0

def append_to_excel(record: dict, path: str = EXCEL_LOG_PATH, sheet: str = "Tahminler"):
    """
    Tek bir kayıtı Excel'e yazar (varsa sonuna ekler).
    'openpyxl' kurulu olmalı. Sorun olursa okur-birleştirir-tekrar yazar.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_df = pd.DataFrame([record])

    # Basit/sağlam yol: dosya varsa oku, birleştir, baştan yaz.
    try:
        if os.path.exists(path):
            try:
                old = pd.read_excel(path, sheet_name=sheet)
            except Exception:
                old = pd.read_excel(path)  # sheet yoksa varsayılanı dene
            out = pd.concat([old, new_df], ignore_index=True)
        else:
            out = new_df

        # Baştan yaz (tek sayfa)
        with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
            out.to_excel(w, index=False, sheet_name=sheet)
        return True, None
    except Exception as e:
        return False, str(e)

def read_all_from_excel(path: str = EXCEL_LOG_PATH, sheet: str = "Tahminler") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path)
    # Zaman alanına göre sırala (varsa)
    if "Zaman" in df.columns:
        df["Zaman_dt"] = pd.to_datetime(df["Zaman"], errors="coerce")
        df = df.sort_values("Zaman_dt", ascending=False).drop(columns=["Zaman_dt"])
    return df

# --------------- Kapak ---------------
def page_home():
    st.title("🧠 Makine Öğrenmesi ile Böbrek Taşı Tahmini")
    img_bytes, src = find_cover_bytes()
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if img_bytes: st.image(img_bytes, use_column_width=True, caption=f"Kapak görseli: {src}")
        else:
            st.warning("Kapak görseli bulunamadı. Aşağıdan yükleyebilirsiniz.")
            up = st.file_uploader("Kapak görseli yükle (jpg/png)", type=["jpg","jpeg","png"], key="up_cover")
            if up is not None:
                st.session_state.cover_bytes = up.getvalue()
                st.image(st.session_state.cover_bytes, use_column_width=True, caption="Yüklenen görsel")

    st.markdown("""
### Klinik karar desteği
Bu uygulama; klinik ve görüntüleme temelli girdilerden **operasyon gereksinimini** tahmin eden,
eğitimli bir makine öğrenmesi modelini kullanır. Sonuçlar hekim değerlendirmesinin **yerine geçmez**;
yalnızca **destekleyicidir**.
""")

    st.divider()
    cL, cR = st.columns([1,1])
    with cL:
        if st.button("🔍 Analiz için tıklayınız", type="primary"):
            st.session_state.page = "analysis"; st.rerun()
    with cR:
        if st.button("🧾 En son tüm sonuçları gör"):
            st.session_state.page = "results"; st.rerun()

# --------------- Analiz ---------------
def page_analysis():
    ensure_model_loaded()
    pipe, classes, feat_cols = st.session_state.pipe, st.session_state.classes, st.session_state.feat_cols

    left, right = st.columns([3,1])
    with left: st.markdown("## 🔎 Analiz")
    with right:
        if st.button("🔄 Modeli yeniden yükle"):
            ensure_model_loaded(reload=True); st.success("Model yeniden yüklendi ✅")

    st.write("**Beklenen giriş sütun(lar)ı:**", feat_cols)
    st.caption(f"💾 Kayıtlar Excel'e yazılıyor → `{EXCEL_LOG_PATH}`")
    st.divider(); st.subheader("📝 Girdi (Tek Kayıt)")

    st.caption("""
- **Cinsiyet:** Kadın=1, Erkek=2
- **Multiple taş varlığı:** Evet=1, Hayır=0
- **Taş konumu (EN AŞAĞIDAKİ TAŞ):** Proksimal üreter=1, Orta üreter=2
- **Hidronefroz derecesi:** 0 veya 1
- **Taş düşürme hikayesi:** Var=1, Yok=0
- **Alfa bloker kullanımı:** Var=1, Yok=0
- **Rim sign varlığı:** Var=1, Yok=0
- **Üreterin anatomik varyasyonları:** Var=1, Yok=0
- **Periüreteral ödem:** Var=1, Yok=0
""")

    with st.form("form_input"):
        # Kimlik
        k1, k2 = st.columns([1.2,2])
        with k1: dosya_no = st.text_input("Dosya No")
        with k2: hasta_adi = st.text_input("Hasta Adı")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cinsiyet = st.selectbox("Cinsiyeti", ["Kadın (1)", "Erkek (2)"])
            vki = st.number_input("VKİ", min_value=0.0, value=25.0, step=0.1)
            yas = st.number_input("YAŞ", min_value=0, value=45, step=1)
            multiple = st.selectbox("Multiple taş varlığı", ["Evet (1)", "Hayır (0)"])
        with c2:
            tas_sayisi = st.number_input("Taş Sayısı", min_value=0, value=1, step=1)
            tas_konumu = st.selectbox("Taş Konumu (EN AŞAĞIDAKİ TAŞ)", ["Proksimal üreter (1)", "Orta üreter (2)"])
            en_buyuk   = st.number_input("En Büyük Taş Boyutu (mm)", min_value=0.0, value=6.0, step=0.1)
            en_yuksek_hu = st.number_input("En Yüksek Taş Dansitesi (HU)", min_value=0, value=700, step=10)
        with c3:
            tas_dusurme = st.selectbox("Taş Düşürme Hikayesi", ["Var (1)", "Yok (0)"])
            hidronefroz = st.selectbox("Hidronefroz", ["0", "1"])
            alfa_bloker = st.selectbox("Alfa Blokör", ["Var (1)", "Yok (0)"])
            rim_sign    = st.selectbox("Rim Sign", ["Var (1)", "Yok (0)"])
        with c4:
            uret_anat = st.selectbox("Üreterin Anatomik varyasyonları", ["Var (1)", "Yok (0)"])
            peri_odem = st.selectbox("Periüreteral Ödem", ["Var (1)", "Yok (0)"])
            uret_kalin = st.number_input("Üreter Kalınlığı (mm)", min_value=0.0, value=5.0, step=0.1)
            idrar_ph   = st.number_input("İdrar PH", min_value=0.0, max_value=14.0, value=6.0, step=0.1)

        submitted = st.form_submit_button("🚀 Tahmin Et")

    if submitted:
        # Model girdisi
        row = {
            "Cinsiyeti": 1 if "Kadın" in cinsiyet else 2,
            "VKİ": float(vki), "YAŞ": int(yas),
            "Multiple taş varlığı": 1 if "Evet" in multiple else 0,
            "Taş Sayısı": int(tas_sayisi),
            "Taş Konumu": 1 if "Proksimal" in tas_konumu else 2,
            "En Büyük Taş Boyutu (mm)": float(en_buyuk),
            "En Yüksek Taş Dansitesi (HU)": int(en_yuksek_hu),
            "Taş Düşürme Hikayesi": 1 if "Var" in tas_dusurme else 0,
            "Hidronefroz": code_from_label(hidronefroz),
            "Alfa Blokör": 1 if "Var" in alfa_bloker else 0,
            "Rim Sign": 1 if "Var" in rim_sign else 0,
            "Üreterin Anatomik varyasyonları": 1 if "Var" in uret_anat else 0,
            "Periüreteral Ödem": 1 if "Var" in peri_odem else 0,
            "Üreter Kalınlığı (mm)": float(uret_kalin),
            "İdrar PH": float(idrar_ph),
        }
        df_in = pd.DataFrame([row])

        # feat_cols hizala
        feat_cols = st.session_state.feat_cols
        miss = [c for c in feat_cols if c not in df_in.columns]
        extra = [c for c in df_in.columns if c not in feat_cols]
        for c in miss: df_in[c] = np.nan
        if extra: df_in = df_in.drop(columns=extra)
        df_in = df_in[feat_cols]

        st.write("**Model girişi (hizalanmış):**"); st.dataframe(df_in)

        # Tahmin
        try:
            y_enc = st.session_state.pipe.predict(df_in)
            pred_label = st.session_state.classes[int(np.asarray(y_enc)[0])]
            st.success(f"🎯 **Tahmin:** {pred_label}")

            top_prob = None
            if hasattr(st.session_state.pipe.named_steps["model"], "predict_proba"):
                proba = st.session_state.pipe.predict_proba(df_in)[0]
                prob_df = pd.DataFrame({"Sınıf": st.session_state.classes, "Olasılık": proba}) \
                            .sort_values("Olasılık", ascending=False).reset_index(drop=True)
                st.subheader("Olasılıklar"); st.dataframe(prob_df)
                top_prob = float(prob_df.iloc[0]["Olasılık"])

            # Oturum kaydı (ekranda kalsın)
            rec = {
                "Zaman": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Dosya No": st.session_state.get("dosya_no", None) or dosya_no,
                "Hasta Adı": st.session_state.get("hasta_adi", None) or hasta_adi,
                "Tahmin": pred_label,
                "En Yüksek Olasılık": top_prob,
                # bazı önemli girdiler:
                "VKİ": row["VKİ"], "YAŞ": row["YAŞ"],
                "Taş Sayısı": row["Taş Sayısı"], "Taş Konumu": row["Taş Konumu"],
                "En Büyük Taş Boyutu (mm)": row["En Büyük Taş Boyutu (mm)"],
                "En Yüksek Taş Dansitesi (HU)": row["En Yüksek Taş Dansitesi (HU)"],
                "Hidronefroz": row["Hidronefroz"],
            }
            st.session_state.records.append(rec)
            st.info("Bu tahmin oturum kaydına eklendi.")

            # Excel'e BAS (otomatik kaydet)
            ok, err = append_to_excel(rec, EXCEL_LOG_PATH)
            if ok:
                st.success(f"💾 Excel'e kaydedildi → `{EXCEL_LOG_PATH}`")
            else:
                st.error(f"Excel'e yazılamadı: {err}")

        except Exception as e:
            st.error(f"Tahmin hatası: {e}")

    # Oturum kayıtları (ekranda kalır) + Excel'i indir (opsiyonel)
    st.divider()
    st.subheader("🗂 Oturum Kayıtları (bu tarayıcı oturumu)")
    if st.session_state.records:
        log_df = pd.DataFrame(st.session_state.records)
        st.dataframe(log_df, use_container_width=True)

        # İstersen indir (Excel'de de var ama kolaylık olsun)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            log_df.to_excel(w, index=False, sheet_name="Tahminler")
        buf.seek(0)
        st.download_button("📥 Bu oturumu Excel indir", buf.getvalue(),
                           file_name="tahminler_oturum.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("Henüz oturum kaydı yok.")

    st.divider()
    cL, cR = st.columns([1,1])
    with cL:
        if st.button("⬅ Kapak sayfasına dön"):
            st.session_state.page = "home"; st.rerun()
    with cR:
        if st.button("🧾 En son tüm sonuçları gör"):
            st.session_state.page = "results"; st.rerun()

# --------------- Tüm Sonuçlar (Excel'den) ---------------
def page_results():
    st.markdown("## 🧾 En Son Tüm Sonuçlar (Excel)")
    df = read_all_from_excel(EXCEL_LOG_PATH)
    if df.empty:
        st.info(f"Excel dosyası bulunamadı ya da boş: `{EXCEL_LOG_PATH}`")
    else:
        st.dataframe(df, use_container_width=True)

        # Excel'i indir (tam dosya)
        with open(EXCEL_LOG_PATH, "rb") as f:
            data = f.read()
        st.download_button("📥 Raporu Excel indir (diskteki dosya)",
                           data, file_name=os.path.basename(EXCEL_LOG_PATH),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    st.divider()
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("⬅ Kapak sayfasına dön"):
            st.session_state.page = "home"; st.rerun()
    with c2:
        if st.button("🔎 Analiz sayfasına dön"):
            st.session_state.page = "analysis"; st.rerun()

# --------------- Router ---------------
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "analysis":
    page_analysis()
else:
    page_results()
