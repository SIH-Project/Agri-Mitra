package com.example.agrimitra;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.squareup.picasso.Picasso;

import java.util.List;

public class NewsAdapter extends RecyclerView.Adapter<NewsAdapter.NewsViewHolder> {

    private final Context context;
    private final List<NewsItem> newsList;
    private TranslationHelper translationHelper;

    public NewsAdapter(Context context, List<NewsItem> newsList) {
        this.context = context;
        this.newsList = newsList;
        this.translationHelper = new TranslationHelper(context);
    }

    @NonNull
    @Override
    public NewsViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.news_item, parent, false);
        return new NewsViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull NewsViewHolder holder, int position) {
        NewsItem news = newsList.get(position);
        
        // Initialize translation and translate text elements
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(holder.tvTitle, news.getTitle());
            translationHelper.translateTextView(holder.tvDesc, news.getDescription());
            translationHelper.translateText("Source: " + news.getSource(), translatedText -> {
                holder.tvSource.setText(translatedText);
            });
            translationHelper.translateText("Published: " + news.getPublishedAt(), translatedText -> {
                holder.tvDate.setText(translatedText);
            });
        });

        if (news.getImageUrl() != null && !news.getImageUrl().isEmpty() && !news.getImageUrl().equals("null")) {
            Picasso.get().load(news.getImageUrl()).into(holder.ivImage);
        }
    }

    @Override
    public int getItemCount() {
        return newsList.size();
    }

    static class NewsViewHolder extends RecyclerView.ViewHolder {
        ImageView ivImage;
        TextView tvTitle, tvDesc, tvSource, tvDate;

        public NewsViewHolder(@NonNull View itemView) {
            super(itemView);
            ivImage = itemView.findViewById(R.id.ivNewsImage);
            tvTitle = itemView.findViewById(R.id.tvNewsTitle);
            tvDesc = itemView.findViewById(R.id.tvNewsDesc);
            tvSource = itemView.findViewById(R.id.tvNewsSource);
            tvDate = itemView.findViewById(R.id.tvNewsDate);
        }
    }
}
