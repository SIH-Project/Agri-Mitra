package com.example.agrimitra;

import android.os.Bundle;
import android.util.Log;
import android.widget.EditText;
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.logging.HttpLoggingInterceptor;

public class ChatBot extends AppCompatActivity {

    private static final String API_URL = "https://openrouter.ai/api/v1/chat/completions";
    private static final String API_KEY = "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"; // 🔹 replace with your OpenRouter key

    private TranslationHelper translationHelper;

    RecyclerView recyclerView;
    EditText etMessage;
    CardView btnSend;
    ProgressBar progressBar;

    BotChatAdapter adapter;

    List<ChatMessage> messages = new ArrayList<>();
    OkHttpClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat_bot);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        recyclerView = findViewById(R.id.recyclerView);
        etMessage = findViewById(R.id.etMessage);
        btnSend = findViewById(R.id.btnSend);
        progressBar = findViewById(R.id.loadingProgress);

        adapter = new BotChatAdapter(this,messages);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);

        // 🔹 OkHttp logging for debugging
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor(message ->
                Log.d("HTTP_LOGS", message));
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);

        client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .build();

        btnSend.setOnClickListener(v -> {
            String userMsg = etMessage.getText().toString().trim();
            if (!userMsg.isEmpty()) {
                addMessage(userMsg, true);
                etMessage.setText("");
                sendMessageToOpenRouter(userMsg);
            }
        });
    }

    private void addMessage(String text, boolean isUser) {
        runOnUiThread(() -> {
            messages.add(new ChatMessage(text, isUser));
            adapter.notifyItemInserted(messages.size() - 1);
            recyclerView.scrollToPosition(messages.size() - 1);
        });
    }

    private void sendMessageToOpenRouter(String prompt) {
        runOnUiThread(() -> progressBar.setVisibility(ProgressBar.VISIBLE));

        MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        JSONObject json = new JSONObject();
        try {
            json.put("model", "openai/gpt-oss-20b:free"); // 🔹 choose model

            JSONArray msgArray = new JSONArray();

            // System prompt → restricts bot to agriculture only
            JSONObject systemMsg = new JSONObject();
            systemMsg.put("role", "system");
            systemMsg.put("content", "You are AgriMitra, an expert agriculture assistant. " +
                    "Answer only questions related to farming, crops, soil, irrigation, fertilizers, " +
                    "pests, government schemes for farmers, and agriculture-related topics. " +
                    "If the user asks anything unrelated to agriculture, reply strictly with: " +
                    "\"Please ask questions related to agriculture.Note: do not give tables and headings in the response only text response\"");
            msgArray.put(systemMsg);

            // User input
            JSONObject userMsg = new JSONObject();
            userMsg.put("role", "user");
            userMsg.put("content", prompt);
            msgArray.put(userMsg);

            json.put("messages", msgArray);

        } catch (JSONException e) {
            e.printStackTrace();
        }

        RequestBody body = RequestBody.create(json.toString(), JSON);

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                runOnUiThread(() -> {
                    progressBar.setVisibility(ProgressBar.GONE);
                    translationHelper.translateText("Error: " + e.getMessage(), translatedText -> {
                        addMessage(translatedText, false);
                    });
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                runOnUiThread(() -> progressBar.setVisibility(ProgressBar.GONE));

                if (!response.isSuccessful()) {
                    runOnUiThread(() -> {
                        translationHelper.translateText("Error: " + response.message(), translatedText -> {
                            addMessage(translatedText, false);
                        });
                    });
                    return;
                }

                String responseData = response.body().string();
                Log.d("OpenRouter_Response", responseData);

                try {
                    JSONObject jsonResponse = new JSONObject(responseData);
                    JSONArray choices = jsonResponse.getJSONArray("choices");
                    JSONObject message = choices.getJSONObject(0).getJSONObject("message");
                    String reply = message.getString("content");

                    addMessage(reply.trim(), false);

                } catch (Exception e) {
                    runOnUiThread(() -> {
                        translationHelper.translateText("Parse error: " + e.getMessage(), translatedText -> {
                            addMessage(translatedText, false);
                        });
                    });
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
