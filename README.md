# canadian_baseball_national_db

## TL;DR
1. Canadian College Baseball has no one stop shop to view performance
2. Use BeautifulSoup To Scrape Data
3. Decompose Data and insert into a PostgreSQL database similar to the Lahman Set
4. Kernel Ridge Regression was able to acheive an RMSE of 0.067 and a R^2 of 0.85 on two different datasets while simple linear regression achieved an R^2 of 0.85 with an RMSE of 0.070
5. Reach out to get involved!

![Linear 1](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/189754ee-b028-40bb-acc8-451c5cb4d0c6)

## Introduction
The Canadian College Baseball Conference is one of three independent collegiate baseball conferences. Starting in 2002, the conference has been a player and community-funded athletic endeavour that has offered the only opportunity for collegiate baseball players in Canada. While not primarily known outside of British Columbia and Alberta (the two provinces that host teams), the conference has hosted many of the top Canadian Baseball players to come out of Canada. While many talented athletes have played in this conference, the goal for many athletes is to move on to an American conference (NCAA or NAIA). Alongside the culture and level of competition, players playing baseball in the US are more likely to sign professional contracts into Major League Baseball (MLB). A considerable barrier to entry is the visibility of players in Canada. Before the internet, travelling to Canada to scout any of these players was economically unfeasible. Therefore, many Canadians wound up stuck in the country. Opportunities have been improving exponentially since the internet. As digital scouting has changed the high school recruiting landscape, more Canadians are getting opportunities in the NCAA. However, these opportunities are still generally only accessible to high school athletes at the top-tier academies. Trevor Brigden is a top prospect of MLB’s best team, the Tampa Bay Rays. Brigden was drafted in the 17th round by Okanagan College (BC) after videos of him playing went viral on Twitter. Better visibility into the statistics would help grow the game in the country and improve the player’s odds of getting drafted by MLB teams. Currently, some of the data is available on a site called pointstreak. The site hosts many junior, high school and college sports statistics. Teams all input stats live during games, typically done by a parent, player, or administrator (if in a more notable conference). The data is only available in team formats and lacks player info, key statistics, rosters, data export, or creative design. There are many missing statistics, and it lacks much flexibility in viewing the data. We aimed to build a preliminary pipeline and machine learning project to predict team winning percentage. Winning percentage is the number of wins a team wins the regular season divided by their total games played. In addition to traditional statistics like runs scored and hits, this paper will build other statistical measures of team performance. In the big picture, this is a proof of concept, and we hope to scale the data scraping and develop a front-end web application that houses these statistics for anyone to view.

## Methodology
### Web Scrapes
Beautiful Soup was used to scrape the data. Six different functions were created to scrape different data types in other table formats, making pandas dataframes and storing data as CSV files. It generally works with one function, scrape page, which is used to scrape a specific page, and some other function used to scrape a particular table, such as scrape batting table. Further work should be done cleaning up the table scrape function. Passing a string into the soup selector caused consistent errors we could not fix. If this error is resolved, this could be cleaned into only two functions, although this framework performs well.

### Database
The best way to store the data was ambiguous. One of the most famous baseball datasets is the Lahman dataset. Created by Sean Lahman, it details all major league baseball players and team
performance. The Lahman dataset is a common training set for regression problems in statistics and machine learning communities. We ultimately elected to model our database with this in mind.
Ultimately the ccbc db has 6 tables: player batting, player pitching, team, team batting, team pitching and standings. Primary keys are often composite composed of the team or player name, the year, and the season type (i.e. PBA, 2017, Regular Season). We did mutate unique IDs for teams in the team table.

![lahman UML](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/bf250376-8ba6-4808-814a-cac28c5e91a6)


### Data Cleaning
The scrapes needed to be merged, decomposed, and reconstructed to create player information by the year. This is the only way to construct player careers. The same general process is done for teams. Unique IDs are built for teams which are then ready for storage. One of the most tedious challenges of this dataset is Different tables have different names for teams. Additionally, the conference plays a highly varied non-conference schedule against American teams when the weather is poor in Canada. This, alongside manual data entry, leads to lots of name key value matching. This was accomplished by using a dictionary manually. Further investigations should be done to match names intuitively. Correctly naming and setting names is crucial for joins in SQL, so this process was necessary to provide accurate data. Finally, we mutated the season type, and the year and seasons were eventually stored with a team full and abbreviated names and ready for storage. Pitching and batting followed a very similar format. There were a few errors from isolated games or players that caused problems. A few examples were dropped from non-conference American teams. Data was ultimately inserted into databases with violation exceptions to catch errors and duplicate entries, which helped us ensure data accuracy.

### Feature Engineering

Next, we used existing data to create extra statistics to model the data on. Overall, nine additional statistics were added from advanced sabermetric libraries. These include simple statistics like on-base percentage and complicated ones like weighted on-base average.

wOBA = (0.69×(BB/AB))+(0.89×(1B/AB))+(1.27×(2B/AB))+(1.62×(3B/AB))+(2.10×(HR/AB))

## EDA

Before modelling, some basic exploratory data analysis was done. Pitching and hitting statistics are generally very similar. Therefore, multicollinearity was a foreseen issue. A correlation heat map was developed to assess how strong this issue may be. The heatmap shows blocks of statistics that are correlated. Essentially most hitting statistics are related, and most pitching statistics are related. This will cause issues in modelling, particularly with methods like the Lasso that struggle with datasets suffering from multicollinearity. 

![Correlation Heatmap](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/0da1323c-c9b9-4b46-9eba-c79dacc518c4)

Considering these results, the dataset appeared a strong candidate for dimensionality reduction. To accomplish this, a Principal Component Analysis (PCA) was executed. PCA reduces the number of dimensions to a smaller number, often used in machine learning pipelines for high dimensionality or colinear predictor variables. Ultimately seven principal components were chosen. Seven dimensions allowed us to explain over 95% of the variance in the dataset.

![PCA](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/f4ef9509-00cc-42fe-8316-a838553651ed)

## Models and Datasets
As mentioned, the modelling goal of this work was to predict team winning percentages. This problem was approached in multiple ways. First, five different regression algorithms were chosen that could potentially model the problem: Kernel Ridge Regression (KRR), Lasso Regression, Gradient Boosting Regressor (GBR), Random Forest Regressor (RF) and Simple Linear Regression. Next, three different datasets were chosen. The first dataset only contained one feature, the team’s run differential, and one target, the winning percentage. Next, a dataset with 27 features (all of our scraped and engineered statistics) was prepared. The dataset was scaled with the StandardScaler, and the target remained the winning percentage. Finally, the last dataset used principal component analysis to reduce the 27 scaled features into seven unique dimensions. Each model is tested on each dataset, besides the simple linear regression, which only uses the run differential, and is inspired by the work in Analyzing Baseball Data with R

## Results

Considering that our target variable is the winning percentage, we use Root Mean Squared Error (RMSE) and R-Squared (R2) to evaluate the performance of our models. RMSE can be viewed as the average difference from the actual winning percentage, while R2 represents the amount of the variance explained. So observing the first row for the Kernel Ridge Regression on the Run Differential dataset, our model is off by an average of 0.067 of winning percentage while explaining 85% of the variance. Overall the models achieved a maximum test R2 of 0.85 and a minimum RMSE of 0.067. The best machine learning algorithm is the Kernel Ridge Regression which has two models tied for the best performance. The only model that performed poorly was the KRR on the reduced dimensions dataset. The simple linear regression, as depicted in 4, performed well, tying the leaders for R2 but slightly lagging behind on RMSE.

![Model Results](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/0f5b5c67-530c-46f9-9720-6ce8873e28b3)

![Linear 2](https://github.com/chrisewanik/canadian_baseball_national_db/assets/113730877/189754ee-b028-40bb-acc8-451c5cb4d0c6)

## Conclusion and Future Work

Overall this project turned out incredibly well. Web scraping was proved feasible and executed quickly. A solid foundation was designed for the PostgreSQL database, and after some tricky data decomposition, many tables could be placed in normalized form. Next, the study accurately models winning percentage and proves that the relationship between run differential and wins is relatively similar to Major League Baseball. The high R^2 values received during the modelling phase indicate that despite the lower level of competition and smaller game schedule, certain relationships that exist in Major League Baseball appear to exist in Canadian College Baseball. The substantial value of run differential also suggests that this could be a reliable metric for seeding teams in a Canadian University World Series. These findings also build upon the questions posed in [2] using RPI for a ranking system. The criteria used in their paper would exclude the Prarie Baseball Academy from Canadian Collegiate World Series. However, by run differential, the 2016 and 2017 teams rank 4th and 1st in the CCBC pointsreak era. This suggests that using other metrics (like Pythagorean expected winning percentage) could be reliable measures for measuring excellence when interconference play is not feasible. It would be interesting to rank teams on other means, like true outcomes (HR, BB, Strikeouts). Homeruns, in particular, are probably a great measure of offensive ability, as they should be somewhat independent of competition (although that is not proven). There is much work to be done in the future. First, the data scrape function could be cleaned up by adding an argument allowing users to specify the HTML element they want on a page. This would allow for quicker web scraping and would generalize better to other tables. Scarpes should be extended to pull any other conference’s data that is made publicly available. There is still a lot of baseball data, rosters are available on websites, and other playing statistics remain available. Sophisticated text scrapes could potentially yield spatial data from the play-by-play entries. More statistics should be generated that are consistent with traditional baseball analysis. While machine learning can solve some baseball analytics problems, most are more designed to be solved statistically. It is unclear whether a conventional SQL database is the correct choice. Given the challenging and lengthy data cleaning, a NoSQL database may be better. Data is generally viewed by either the team or the player, which could make NoSQL a good choice. Finally, this data could be wonderfully presented in some analytics dashboards. Now that this data is stored and there appears to be some reliable collection method
increasing visibility is a huge goal. Some of the schools in the east, such as the University of Toronto,
University of Waterloo and Universit´e de. Montr´eal are famous for its computer science, statistics
and data science programs. It seems likely that at least one or two baseball players are interested
in the preliminary work this paper sets out. Leveraging talent across the country, building a better
relationship, and open-sourcing parts of this project could increase the growth speed exponentially
and do significant good in growing the game in Canada.

## Get Involved

If you have interest in contributing to this project (even if you have very little experience!) please reach out! The main goal of this project is to develop a system that grows the game in Canada. In the odd chance you are a coach or administrator (with no desire to program), still please contact us! We desperately need more contacts in eastern Canada and would love to grow this project to become the one stop shop for baseball statistics in Canada. 


