package com.example.agrimitra;

public class NewsItem {
    private final String title;
    private final String description;
    private final String source;
    private final String publishedAt;
    private final String imageUrl;

    public NewsItem(String title, String description, String source, String publishedAt, String imageUrl) {
        this.title = title;
        this.description = description;
        this.source = source;
        this.publishedAt = publishedAt;
        this.imageUrl = imageUrl;
    }

    public String getTitle() { return title; }
    public String getDescription() { return description; }
    public String getSource() { return source; }
    public String getPublishedAt() { return publishedAt; }
    public String getImageUrl() { return imageUrl; }
}
