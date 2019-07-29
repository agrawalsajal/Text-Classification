from final_project import *


with open('TFIDF.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('svcclassifier.pickle','rb') as f:
    clf = pickle.load(f)
    
    
# sample = ['''
# The point is that government has made the policy, 
# but there is no one to make sure that it is reaching 
# those who actually need to be benifitted from this policy. 
# So, what  my small point is that government should farme certain policy to make these existing policies better. 
# Otherwise this country will remain in this state of mess the it has been for past 70 years. 
# There wouldn't be any difference in the way Britishers left it to way we are now.''']

# sample = '''First of all..a v big sympathy for the haters of this movie..I'm really shocked at the cheap mental and aesthetic level of those people who are finding faults and criticizing this Fantabulous movie..

# To be v v honest, I'm not much of a person to watch multi-starrer movies as Im more of a lover of romantic movies with one couple nd story around them..But after watching this movie, I felt completely absorbed in the magnificent direction, artistic skills all around, stunning acting by all the mega stars, the each and every bit of grandeur shown..

# I, sincerely say that it's the most Wonderful piece of Art I have ever watched in my life..I felt soo amazed to have such talent in Bollywood and really I felt proud to have such fabulous moments in my life to watch such Master Piece of Art..

# Madhuri Dixit, Varun Dhawan, Alia Bhatt, Sanjay Dutt, Sonakshi Sinha, Aditya Roy Kapoor---- I'm really speechless for the wonderful roles they played and whatever role was theirs, they did absolute justice to it.. The roles of Madhuri, Varun and Alia are highly highly commendable..My big appreciation and admiration to such stars who played the characters soo well and put life in them as real..

# I will give 4.5 stars out of 5💝💝💝💝

# I'm so happy to see the fantabulous Picturisation and Cinematography in the songs played on Madhuri and Alia..
# Will watch again soon💟💟💟💟💟💟'''

sample = '''
    So Called Epic Love Saga turned out EPIC DISAPPOINTMENT 1 Star ⭐️ 
Even if you watch this movie at 5x speed it never ends,3 Hours Long Disaster,
 hard to survive through this even 1 hour despite having such promising huge cast , 
 larger than life sets and humongous budget which is shelled out on everything other than script.
 Bunch of talented actors but none have a scope for anything nor had proper screentime,
 Songs pop up out of nowhere and has no relevace to the happenings in movie,
 Screenplay is lous																																																																																														y dragged and repetitive. 
 Only saving grace which fetch this movie a star is larger than life presentation and huge lit sets .
  A movie that you will be glad of missing in the cinemas 
'''

# sample = '''
# 	Honestly i could not be happier with the outcomes of this movie,
# 	 now sorry for saying this but this review will contain spoilers.
# 	  Seeing some of my favourite MCU characters come to an end including 
# 	  Thanos was a real depressing moment, and left me and everyone feeling a
# 	   bit unfilled, not because the movie didnt exceed expectations.....
# 	   its because the movie left of with nothing regarding the future of the MCU.
# 	    Yes Thor is seen with the Guardians Of The Galaxy, but we knew that the Guardians 
# 	    have an upcoming sequel and to be honest, the Avengers are the pinacle of hype and 
# 	    excitment towards the MCU, and with 3 of the 6 "OG" avengers actually dying them being
# 	     Iron Man "Tony Stark",Captain America"Steve Rogers" and Black Widow "Natasha Romenoff"
# 	      it left the audience feeling sustained but still hungry for some more footage or any 
# 	      grasp of seeing the Main Avengers. We still do have many more movies regarding other
# 	  		characters such as Spiderman, Black Panther and Guardians of the Galaxy but i still will
# 	        never be even close to Excited and Hyped to see those movies as much as Avengers Endgame.
# 	         I also wanted to complement the screentime of alot of our characters, the Russo Brothers 
# 	         did this so perfectly and i believe that they should be in charge of alot more of the.
# 	          future of the Marvel Cinematic Universe. Regarding other Hated/Less Loved characters 
# 	          such as Captain Marvel and Doctor Strange, im glad they didnt give much screentime because 
# 	          i dont think anyone genuinlly likes seeing these guys hog up any of the 3 hours that the
# 	           movie was and im happy to see that her power wasnt enough to even fuss Thanos. 
# 	           Overall i would give this movie a 10/10, its excellent and well blended together 
# 	           and every moment of the 3 hours that i watched the movie was just so amazing,not to 
# 	           mention the pure atmosphere of the Cinema at the time was just so heartwarming, 
# 	           and i thank everyone that has ever been involved with the MCU for making a brilliant 
# 	           and interesting franchise that has made peoples childhood such a memorable phenomenom. 
# 	           Stan Lee you are an amazing and creative man, you'll be remembered as such a legend and 
# 	           i hope every moment of your life was as amazing as you made mine when i watched the ending 
# 	           to the worlds most beloved franchise R.I.P.
# '''

sample = nltk.sent_tokenize(sample)

sample = process_data(sample, missed_words, stop_words)
sample = tfidf.transform(sample).toarray()
sentiment = clf.predict(sample)
print(sentiment.mean()*5)

