# Taxi

## Implementation
* Problem solving in Open AI Gym Taxi-v3 environment
* I use Open Al Gym.

```
    "There are 4 locations (labeled by different letters), and our job is to pick up the
    passenger at one location and drop him off at another. We receive +20 points for a
    successful drop-off and lose 1 point for every time-step it takes. There is also a -10 point
    as the penalty for illegal pick-up and drop-off actions."

    https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
```


## Order of Visits and Problem to solve
The Action Space = 6
[0. down, 1. Up, 2. Right, 3. Left, 4. Pickup, 5. Dropoff]

Passengers wait at points R, G, Y, B and move up, down, left and right.
After moving to the location, you must pick up and then go to the destination and drop off.

reward is -1, -10, or 20
done is True when 20 points are guaranteed by dropping off passengers at the correct destination, or True when 200 steps are exceeded, and False otherwise.
