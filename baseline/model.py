'''
Modeling users, interactions and items from
the recsys challenge 2017.

by Daniel Kohlsdorf
'''


class User:
    def __init__(self, title, clevel, indus, disc, country, region, premium, numb_work_experiences, years_experience,
                 years_experience_current, edu_degree, edu_fieldofstudies, wtcj):
        self.title   = title
        self.clevel  = clevel
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region
        self.premium = premium
        self.numb_work_experiences = numb_work_experiences
        self.years_experience = years_experience
        self.years_experience_current = years_experience_current
        self.edu_degree = edu_degree
        self.edu_fieldofstudies = edu_fieldofstudies
        self.wtcj = wtcj

    def __repr__(self):
        return "\nthis user:{},{},{},{},{},{},{} \nends here\n"\
            .format(self.title, self.clevel, self.indus, self.disc, self.country, self.region, self.premium)


class Item:
    def __init__(self, title, clevel, indus, disc, country, region):
        self.title   = title
        self.clevel  = clevel
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region


class Interaction:
    def __init__(self, user, item, interaction_type):
        self.user = user
        self.item = item
        self.interaction_type = interaction_type

    @staticmethod
    def save(interactions, file):
        import csv
        users_keys = list(['user_{}'.format(key) for key in interactions[0].user.__dict__.keys()])
        item_keys = list(['item_{}'.format(key) for key in interactions[0].item.__dict__.keys()])
        header = users_keys + item_keys + ['interaction_type']
        print(type(header))

        with open(file, 'w') as output_file:
            csv_writer = csv.writer(output_file, delimiter=',')
            print(type(header))
            csv_writer.writerow(header)

            for i in interactions:
                user_values = list(i.user.__dict__.values())
                item_values = list(i.item.__dict__.values())
                csv_writer.writerow(user_values + item_values + [i.interaction_type])

    def title_match(self):
        return float(len(set(self.user.title).intersection(set(self.item.title))))

    def clevel_match(self):
        if self.user.clevel == self.item.clevel:
            return 1.0
        else:
            return 0.0

    def indus_match(self):
        if self.user.indus == self.item.indus:
            return 1.0
        else:
            return 0.0

    def discipline_match(self):
        if self.user.disc == self.item.disc:
            return 2.0
        else:
            return 0.0

    def country_match(self):
        if self.user.country == self.item.country:
            return 1.0
        else:
            return 0.0

    def region_match(self):
        if self.user.region == self.item.region:
            return 1.0
        else:
            return 0.0

    def premium(self):
        return self.user.premium

    def features(self):
        return [
            self.title_match(), self.clevel_match(), self.indus_match(), 
            self.discipline_match(), self.country_match(), self.region_match(),
            self.premium()
        ]

    def label(self): 
        if self.interaction_type == 4: 
            return 0.0
        else:
            return 1.0


