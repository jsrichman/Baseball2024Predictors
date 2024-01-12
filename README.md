# Baseball Research on the Top Influence Players for 2024
# By Jordan S Richman
# Project started on 11/1/2023

# Definition: Expected Weighted On-base Average (xwOBA) is formulated using exit velocity, launch angle and,
# on certain types of batted balls, Sprint Speed. In the same way that each batted ball is assigned an expected
# batting average, every batted ball is given a single, double, triple, and home run probability based on the results
# of comparable batted balls since StatCast was implemented Major League-wide in 2015. For the majority of batted
# balls, this is achieved using only exit velocity and launch angle. As of 2019, "topped" or "weakly hit" balls also
# incorporate a batter's seasonal Sprint Speed. All hit types are valued in the same fashion for xwOBA as they are in
# the formula for standard wOBA: (unintentional BB factor x unintentional BB + HBP factor x HBP + 1B factor x 1B + 2B
# factor x 2B + 3B factor x 3B + HR factor x HR)/(AB + unintentional BB + SF + HBP), where "factor" indicates the
# adjusted run expectancy of a batting event in the context of the season as a whole. Knowing the expected outcomes
# of each individual batted ball from a particular player over the course of a season -- with a player's real-world
# data used for factors such as walks, strikeouts, and times hit by a pitch -- allows for the formation of said
# player's xwOBA based on the quality of contact, instead of the actual outcomes. Likewise, this exercise can be done
# for pitchers to get their expected xwOBA against.

# Why it's useful xwOBA is more indicative of a player's skill than regular wOBA, as xwOBA removes defense from the
# equation. Hitters, and likewise pitchers, are able to influence exit velocity and launch angle but have no control
# over what happens to a batted ball once it is put into play.

# Below 0.300 is considered poor
# 0.320 is considered average
# 0.350 is considered good
# 0.400 is considered excellent
