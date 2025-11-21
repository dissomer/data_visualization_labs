import matplotlib.pyplot as plt
import networkx as nx
from instagrapi import Client
from instagrapi.exceptions import TwoFactorRequired

MAX_FOLLOWINGS_COUNT = 20

instagram_client = Client()
instagram_client.delay_range = [1, 5]

USERNAME = input("input user name:")
PASSWORD = input("input user password:")

assert USERNAME, 'LOGIN should be inputed'
assert PASSWORD, 'PASSWORD should be inputed'

try:
    instagram_client.login(USERNAME, PASSWORD)
    print("Logged in successfully")
except TwoFactorRequired:
    print("Two-factor authentication required. Please disable it in your Instagram settings.")
    raise

my_followings = instagram_client.user_following(
    user_id=instagram_client.user_id,
    amount=MAX_FOLLOWINGS_COUNT
)

my_followings_names = [user.username for user in my_followings.values()]

G = nx.Graph()
G.add_node(instagram_client.username, label=instagram_client.username)

for following in my_followings.values():
    G.add_node(following.username, label=following.full_name)
    G.add_edge(instagram_client.username, following.username)

for person in my_followings.values():
    try:
        print(f'Processing following person: [{person.username}] followings...')
        following_followings = instagram_client.user_following(person.pk)

        for following in following_followings.values():
            if following.username in my_followings_names:
                G.add_node(following.username, label=following.full_name)
                G.add_edge(person.username, following.username)

    except Exception as e:
        print(f"Error fetching data for {person.username}: {e}")

print('Saving graph...')
nx.write_gexf(G, "InstaFriends.gexf")

print("Drawing...")
plt.figure(figsize=(12, 10))
nx.draw_spring(G, with_labels=True, font_weight='bold', font_size=5)
plt.savefig('InstaGraf.png', dpi=600)
plt.show()

if nx.is_connected(G):
    diameter = nx.diameter(G)
    print("Diameter of the graph:", diameter)
else:
    print("Graph is not connected")

