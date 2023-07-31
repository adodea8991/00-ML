import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import messagebox

def scrape_tripadvisor_description(url):
    # Check if the URL is from TripAdvisor
    if "tripadvisor.com/" not in url:
        messagebox.showinfo("Error", "The provided URL is not from TripAdvisor.")
        return

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        # Fetch the webpage content
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the div with the specified class
        description = soup.find("div", class_="biGQs _P pZUbB KxBGd")

        if description:
            return description.text.strip()
        else:
            return "Description not found."

    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return

if __name__ == "__main__":
    # Replace the 'your_url_here' with the TripAdvisor URL you want to scrape
    url_to_scrape = input("Please give a URL: ")

    # Scrape and display the TripAdvisor description
    description = scrape_tripadvisor_description(url_to_scrape)
    print(description)
