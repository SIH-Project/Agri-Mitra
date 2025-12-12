package com.example.agrimitra;

import android.util.Log;

import com.google.mlkit.nl.translate.Translator;
import com.google.mlkit.nl.translate.TranslatorOptions;
import com.google.mlkit.nl.translate.Translation;

import androidx.annotation.NonNull;

public class MLKitTranslator {

    private Translator translator;

    // Initialize translator with source and target languages
    public void initTranslator(String sourceLangCode, String targetLangCode) {
        TranslatorOptions options = new TranslatorOptions.Builder()
                .setSourceLanguage(mapToMLKitLang(sourceLangCode))
                .setTargetLanguage(mapToMLKitLang(targetLangCode))
                .build();
        translator = Translation.getClient(options);
    }

    // Download model automatically (no network/charging restrictions)
    public void downloadModelIfNeeded(Runnable onSuccess, Runnable onFailure) {
        translator.downloadModelIfNeeded()
                .addOnSuccessListener(aVoid -> {
                    Log.d("MLKit", "Model downloaded successfully");
                    onSuccess.run();
                })
                .addOnFailureListener(e -> {
                    Log.e("MLKit", "Model download failed", e);
                    onFailure.run();
                });
    }

    // Translate text via callback
    public void translateText(@NonNull String text, @NonNull TranslationCallback callback) {
        translator.translate(text)
                .addOnSuccessListener(callback::onTranslated)
                .addOnFailureListener(e -> {
                    callback.onTranslated(text); // fallback to original text
                    Log.e("MLKit", "Translation failed for text: " + text, e);
                });
    }

    // Close translator to free resources
    public void close() {
        if (translator != null) {
            translator.close();
        }
    }

    // Callback interface
    public interface TranslationCallback {
        void onTranslated(String translatedText);
    }

    // Map standard language codes to ML Kit constants
    private String mapToMLKitLang(String code) {
        switch (code) {
            case "en": return com.google.mlkit.nl.translate.TranslateLanguage.ENGLISH;
            case "hi": return com.google.mlkit.nl.translate.TranslateLanguage.HINDI;
            case "ta": return com.google.mlkit.nl.translate.TranslateLanguage.TAMIL;
            case "te": return com.google.mlkit.nl.translate.TranslateLanguage.TELUGU;
            default: return com.google.mlkit.nl.translate.TranslateLanguage.ENGLISH;
        }
    }
}
