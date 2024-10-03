import argparse
import subprocess

def main():

    parser = argparse.ArgumentParser(description="Script to ask for user's favorite book.")
    parser.add_argument('--script', type=str, default='recommender.py', help='Python script')
    args = parser.parse_args()

    # Ask the user for their favorite book name
    favorite_book = input("What is your number one favorite book you've read (just one book)? ")

    # Run the specified brand recommender script with the favorite book name as an argument
    result = subprocess.run(['python', args.script, favorite_book], capture_output=True, text=True)

    # Print the output from the secondary script
    print(f'Given your favorite book, {favorite_book}, the top three American clothing brands that match the most with your personality are {result.stdout}')

if __name__ == "__main__":
    main()