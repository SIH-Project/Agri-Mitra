package com.example.agrimitra;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RelativeLayout;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NewsPage extends AppCompatActivity {

    private TranslationHelper translationHelper;

    EditText etKeyword;
    Button btnFetchNews;

    RelativeLayout loading;
    RecyclerView rvNews;
    private final String API_KEY = "070a0bc217664f0daa0a7f7660b8366f";

    List<NewsItem> newsList;
    NewsAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_news_page);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        loading=findViewById(R.id.loading);
        loading.setVisibility(View.VISIBLE);
        rvNews = findViewById(R.id.rvNews);

        newsList = new ArrayList<>();
        adapter = new NewsAdapter(this, newsList);
        rvNews.setLayoutManager(new LinearLayoutManager(this));
        rvNews.setAdapter(adapter);

        fetchNews("Agriculture");



        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void fetchNews(String keyword) {
        String url = "https://newsapi.org/v2/everything?q=" + keyword + "&pageSize=5";

        RequestQueue queue = Volley.newRequestQueue(this);

        JsonObjectRequest request = new JsonObjectRequest(Request.Method.GET, url, null,
                response -> {
                    try {
                        if (response.getString("status").equals("ok")) {
                            JSONArray articles = response.getJSONArray("articles");
                            newsList.clear();
                            for (int i = 0; i < articles.length(); i++) {
                                JSONObject news = articles.getJSONObject(i);
                                String title = news.getString("title");
                                String desc = news.getString("description");
                                String source = news.getJSONObject("source").getString("name");
                                String date = news.getString("publishedAt");
                                String imageUrl = news.getString("urlToImage");
                                newsList.add(new NewsItem(title, desc, source, date, imageUrl));
                            }
                            loading.setVisibility(View.GONE);
                            adapter.notifyDataSetChanged();
                        }
                    } catch (Exception e) {
                        loading.setVisibility(View.GONE);
                        Log.e("NewsActivity", "Parsing error", e);
                    }
                },
                error -> {
                    Log.e("NewsActivity", "Volley error: " + error.toString());
                    translationHelper.showTranslatedToast("Failed to fetch news");
                }) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("X-Api-Key", API_KEY);
                headers.put("User-Agent", "Mozilla/5.0");
                return headers;
            }
        };

        queue.add(request);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
