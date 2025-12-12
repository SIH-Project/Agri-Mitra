package com.example.agrimitra;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper class for managing translations across the app
 * Provides common translation operations for Activities and Fragments
 */
public class TranslationHelper {
    
    private MLKitTranslator translator;
    private String selectedLanguage;
    private Context context;
    
    public TranslationHelper(Context context) {
        this.context = context;
        this.translator = new MLKitTranslator();
        
        // Load saved language preference
        SharedPreferences prefs = context.getSharedPreferences("AppPrefs", Context.MODE_PRIVATE);
        this.selectedLanguage = prefs.getString("SelectedLanguage", "en");
        
        // Initialize translator (English to selected language)
        if (!selectedLanguage.equals("en")) {
            translator.initTranslator("en", selectedLanguage);
        }
    }
    
    /**
     * Initialize translation and download model if needed
     * @param onReady Callback when translation is ready
     */
    public void initializeTranslation(Runnable onReady) {
        if (selectedLanguage.equals("en")) {
            // No translation needed for English
            onReady.run();
            return;
        }
        
        translator.downloadModelIfNeeded(
            () -> {
                Log.d("TranslationHelper", "Translation model ready for: " + selectedLanguage);
                onReady.run();
            },
            () -> {
                Log.e("TranslationHelper", "Failed to download translation model");
                // Continue anyway with English fallback
                onReady.run();
            }
        );
    }
    
    /**
     * Translate a single TextView
     * @param textView The TextView to translate
     * @param originalText The original English text
     */
    public void translateTextView(TextView textView, String originalText) {
        if (selectedLanguage.equals("en")) {
            textView.setText(originalText);
            return;
        }
        
        translator.translateText(originalText, translatedText -> {
            textView.setText(translatedText);
        });
    }
    
    /**
     * Translate multiple TextViews at once
     * @param textViews List of TextViews to translate
     * @param originalTexts List of original English texts (same order as TextViews)
     */
    public void translateTextViews(List<TextView> textViews, List<String> originalTexts) {
        if (selectedLanguage.equals("en")) {
            for (int i = 0; i < textViews.size() && i < originalTexts.size(); i++) {
                textViews.get(i).setText(originalTexts.get(i));
            }
            return;
        }
        
        for (int i = 0; i < textViews.size() && i < originalTexts.size(); i++) {
            final int index = i;
            translator.translateText(originalTexts.get(i), translatedText -> {
                textViews.get(index).setText(translatedText);
            });
        }
    }
    
    /**
     * Show a translated Toast message
     * @param originalText The original English text
     */
    public void showTranslatedToast(String originalText) {
        if (selectedLanguage.equals("en")) {
            Toast.makeText(context, originalText, Toast.LENGTH_SHORT).show();
            return;
        }
        
        translator.translateText(originalText, translatedText -> {
            Toast.makeText(context, translatedText, Toast.LENGTH_SHORT).show();
        });
    }
    
    /**
     * Show a translated Toast message with custom duration
     * @param originalText The original English text
     * @param duration Toast duration (Toast.LENGTH_SHORT or Toast.LENGTH_LONG)
     */
    public void showTranslatedToast(String originalText, int duration) {
        if (selectedLanguage.equals("en")) {
            Toast.makeText(context, originalText, duration).show();
            return;
        }
        
        translator.translateText(originalText, translatedText -> {
            Toast.makeText(context, translatedText, duration).show();
        });
    }
    
    /**
     * Translate text and return via callback
     * @param originalText The original English text
     * @param callback Callback to receive translated text
     */
    public void translateText(String originalText, MLKitTranslator.TranslationCallback callback) {
        if (selectedLanguage.equals("en")) {
            callback.onTranslated(originalText);
            return;
        }
        
        translator.translateText(originalText, callback);
    }
    
    /**
     * Get the currently selected language code
     * @return Language code (en, hi, ta, te)
     */
    public String getSelectedLanguage() {
        return selectedLanguage;
    }
    
    /**
     * Check if translation is needed (i.e., selected language is not English)
     * @return true if translation is needed
     */
    public boolean isTranslationNeeded() {
        return !selectedLanguage.equals("en");
    }
    
    /**
     * Close the translator to free resources
     */
    public void close() {
        if (translator != null) {
            translator.close();
        }
    }
}
