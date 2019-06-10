from BeautifulSoup import BeautifulSoup
import sys
import os
import json
import progressbar
import argparse


def get_options():
    parser = argparse.ArgumentParser(description='Data Generator')

    parser.add_argument('-country', action="store", default='Europe', dest='country', type=str)

    parser.add_argument('-data_dir', action="store",
                        default='/scratch/cse/btech/cs1130773/BTP/Data_from_Poojan/Danish_Data/Raw_Data/',
                        dest="data_dir", type=str)
    parser.add_argument('-save_prefix', action="store", default="/scratch/cse/btech/cs1130773/BTP/",
                        dest="save_prefix", type=str)
    options = parser.parse_args(sys.argv[1:])
    return options


def getFirstPostData(forum_text):
    soup = BeautifulSoup(forum_text)
    title = ""
    date = ""
    body = ""
    try:
        date = soup.find("div", attrs={"class": "postDate"}).text
    except AttributeError:
        print("Date not found")
    try:
        title = soup.find("div", attrs={"class": "postTitle"}).text
    except AttributeError:
        print("Title not found")
    try:
        body = soup.find("div", attrs={"class": "postBody"}).text
    except AttributeError:
        print("Body not found, now this is weird")
    return [title, date, body]


def getAllPostData(forum_text):
    soup = BeautifulSoup(forum_text)
    titles = []
    dates = []
    bodies = []
    cleantitles = []
    cleandates = []
    cleanbodies = []

    try:
        dates = soup.findAll("div", attrs={"class": "postDate"})
        # print(dates)
        for date in dates:
            cleandates.append(date.string)
            # print(cleandates)
        # soup.findAll("div", attrs={"class": "postDate"})
    except AttributeError:
        print("Date not found")
    try:
        titles = soup.findAll("div", attrs={"class": "postTitle"})
        # print(titles)
        for title in titles:
            cleantitles.append(title.text)
            # print(cleantitles)
    except AttributeError:
        print("Title not found")
    try:
        bodies = soup.findAll("div", attrs={"class": "postBody"})
        for body in bodies:
            cleanbodies.append(body.text)
            # print(cleanbodies)
    except AttributeError:
        print("Body not found, now this is weird")
    return [cleantitles, cleandates, cleanbodies]


def getMetaData(first_line):
    return first_line.strip().split(',')


def extract_answers(cleantitles, cleandates, cleanbodies):
    title = ""
    body = ""
    date = ""
    answers = {}
    answers['answers'] = []
    if len(cleandates) > 0:
        title = cleantitles[0]
        body = cleanbodies[0]
        date = cleandates[0]
        answers = {}
        answers['answers'] = []
        for i in range(1, len(cleandates)):
            answer = {}
            answer['title'] = cleantitles[i]
            answer['body'] = cleanbodies[i]
            answer['date'] = cleandates[i]
            answers['answers'].append(answer)
    return [title, date, body, answers]


# directory = where all the state wise articles are present
def buildJSON(options):
    done_set = set(['Finland', 'Cyprus', 'Belarus', 'Croatia', 'Czech Republic', 'Austria', 'South Carolina',
                    'Albania', 'Estonia', 'Faroe Islands', 'Bosnia and Herzegovina', 'Belgium',
                    'New York', 'Ohio', 'Virginia', 'Washington', 'Oregon'])
    # revisit_set = set(['Austria', 'South Carolina', 'Belgium', 'Oregon'])
    directory = options.data_dir
    country = directory.split('/')[-1]
    save_prefix = options.save_prefix + country
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    struct = [x for x in os.walk(directory)][1:]
    for i, s in enumerate(struct):
        location = s[0]
        state = location.split('/')[-1]
        if state in done_set:
            continue
        forum_dict = {'posts': []}
        print('DOING FOR STATE', state, ' : (', i + 1, 'of ', len(struct), ')')
        bar = progressbar.ProgressBar(max_value=len(s[2]))
        for ii, file_name in enumerate(s[2]):
            f = open(f"{location}/{file_name}", 'r')
            try:
                [location3, link] = getMetaData(f.readline())
                # print(location3, link)
            except Exception as e:
                print(file_name, "empty")
                continue
            forum_text = ""
            for line in f.readlines():
                forum_text += line + '\n'
            [cleantitles, cleandates, cleanbodies] = getAllPostData(forum_text)

            location1 = country
            location2 = state
            [title, date, body, answers] = extract_answers(cleantitles, cleandates, cleanbodies)

            temp_dict = {'location1': location1,
                         'location2': location2,
                         'location3': location3,
                         'link': link,
                         'title': title,
                         'date': date,
                         'body': body,
                         'answers': answers['answers']}
            forum_dict['posts'].append(temp_dict)
            bar.update(ii)
            sys.stdout.flush()
        save_file_dir = save_prefix + '/' + state
        if not os.path.exists(save_file_dir):
            os.makedirs(save_file_dir)
        save_file_name = save_file_dir + '/forum_posts_answers.json'
        json.dump(forum_dict, open(save_file_name, 'wb'))
        print(state, 'DONE')
        sys.stdout.flush()


if __name__ == '__main__':
    options = get_options()
    options.data_dir = options.data_dir + options.country
    buildJSON(options)
    
