package com.example.agrimitra;

public class MarketPrice {
    private String commodity;
    private String city;
    private String minPrice;
    private String maxPrice;
    private String modelPrice;
    private String date;

    public MarketPrice(String commodity, String city, String minPrice, String maxPrice, String modelPrice, String date) {
        this.commodity = commodity;
        this.city = city;
        this.minPrice = minPrice;
        this.maxPrice = maxPrice;
        this.modelPrice = modelPrice;
        this.date = date;
    }

    public String getCommodity() { return commodity; }
    public String getCity() { return city; }
    public String getMinPrice() { return minPrice; }
    public String getMaxPrice() { return maxPrice; }
    public String getModelPrice() { return modelPrice; }
    public String getDate() { return date; }
}

