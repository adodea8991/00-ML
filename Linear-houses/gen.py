import random
import csv

def generate_house_data(num_points):
    house_size_min = 800
    house_size_max = 3500
    house_price_min = 150000
    house_price_max = 700000

    with open('house_data_generated.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['House Size', 'House Price'])

        for _ in range(num_points):
            house_size = random.randint(house_size_min, house_size_max)
            house_price = random.randint(house_price_min, house_price_max)
            writer.writerow([house_size, house_price])

if __name__ == '__main__':
    num_data_points = 15000
    generate_house_data(num_data_points)
