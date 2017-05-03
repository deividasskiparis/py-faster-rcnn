import numpy as np


class Group(object):

    def __init__(self, groupID):
        self.members = []
        self.memberIDs = []
        self.groupID = groupID

    def __add__(self, other):
        if isinstance(other, Person):
            self.members.append(other)
            self.memberIDs.append(other.personID)

    def __str__(self):
        f = []
        for i in self.members:
            f += [i.personID]
        return "Group %d:\n\tSize: %d\n\tMember IDs: %s\n" % (self.groupID, len(self.members), str(f))

    def __getitem__(self, key):
        return self.members[self.memberIDs.index(key)]
    pass


class Person(object):

    def __init__(self, **kwargs):

        # Default parameters
        self.appearances = 0  # Number of times the person has been detected
        self.last_seen = 0  # Number of time steps since last detection
        self.visibility = []
        self.next_position = None

        # # Kalman
        # self.klm = Kalman_Filter(F=np.matrix('''
        #                   1. 0. 0. 1. 0. 0. 1. 0. 0.;
        #                   0. 1. 0. 0. 1. 0. 0. 1. 0.;
        #                   0. 0. 1. 0. 0. 1. 0. 0. 1.;
        #                   0. 0. 0. 1. 0. 0. 1. 0. 0.;
        #                   0. 0. 0. 0. 1. 0. 0. 1. 0.;
        #                   0. 0. 0. 0. 0. 1. 0. 0. 1.;
        #                   0. 0. 0. 0. 0. 0. 1. 0. 0.;
        #                   0. 0. 0. 0. 0. 0. 0. 1. 0.;
        #                   0. 0. 0. 0. 0. 0. 0. 0. 0.
        #                   '''),
        #                          H=np.matrix('''
        #                   1. 0. 0. 0. 0. 0. 0. 0. 0.;
        #                   0. 1. 0. 0. 0. 0. 0. 0. 0.;
        #                   0. 0. 1. 0. 0. 0. 0. 0. 0.'''))


        # Optional parameters

        # Kalman
        self.klm = Kalman_Filter(F=np.matrix('''
                          1. 0. 0. 0.5 0. 0.;
                          0. 1. 0. 0. 0.5 0.;
                          0. 0. 1. 0. 0. 0.5;
                          0. 0. 0. 1. 0. 0.;
                          0. 0. 0. 0. 1. 0.;
                          0. 0. 0. 0. 0. 1.
                          '''),
                                 H=np.matrix('''
                          1. 0. 0. 0. 0. 0.;
                          0. 1. 0. 0. 0. 0.;
                          0. 0. 1. 0. 0. 0.'''))
        if 'personID' in kwargs.keys():
            self.personID = kwargs['personID']  # [Optional] Person ID
        else:
            self.personID = -1

        # Required Parameters
        try:
            self.fv = np.array([kwargs['feat_vec']])  # [Required] feature vector representing the person
            self.bbox = np.array(kwargs['bbox'])  # [Required] bounding box of detection
            assert self.bbox.shape[0] == 4 and len(self.bbox.shape) == 1

        except:
            raise ValueError("Must provide 'feat_vec' and 'bbox' values")

    def predict_next(self):
        self.next_position = self.klm.predict_next(self.fv[-1, :3])[:,:3]
        return self.next_position

    def __add__(self, other):
        n_mov_avg = min(3, self.fv.shape[0])
        if isinstance(other, Person):

            # mov_avg = np.array([self.fv[n_mov_avg - 1, :]])
            # mov_avg = np.append(mov_avg, other.fv, axis=0)
            self.fv = np.append(self.fv, other.fv, axis=0)

            self.bbox = other.bbox
            other.personID = self.personID

        elif isinstance(other, np.ndarray):
            assert self.fv.shape[1] == other.shape[1] and len(other.shape) == 2
            # mov_avg = np.array([self.fv[n_mov_avg - 1, :]])
            # mov_avg = np.append(mov_avg, other, axis=0)
            self.fv = np.append(self.fv, other, axis=0)

        else:
            raise TypeError("Addition between %s and %s data types does not exist" % (type(self), type(other)))

        return self

    def __str__(self):
        return "Person:\n\tID: %d\n\tFeature Vector: %s\n" % (self.personID, str(self.fv))

class Kalman_Filter():

    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''

    def __init__(self, **kwargs):
        ################################# REQUIRED ##################################

        try:
            self.F = kwargs['F']
            self.H = kwargs['H']
        except:
            raise ValueError("Missing required input values")

        self.dims = self.F.shape[0]

        ################################# OPTIONAL ##################################

        if 'motion' in kwargs.keys():
            self.motion = kwargs['motion']
        else:
            self.motion = np.asmatrix(np.zeros(shape=self.dims, dtype=np.float)).T


        if 'R' in kwargs.keys():
            self.R = kwargs['R']  # [Optional] Person ID
        else:
            self.R = 0.01 ** 2

        if 'Q' in kwargs.keys():
            self.Q = kwargs['Q']  # [Optional] Person ID
        else:
            self.Q = np.matrix(np.eye(self.dims))

        if 'x' in kwargs.keys():
            self.x = kwargs['x']  # [Optional] Person ID
        else:
            self.x = np.asmatrix(np.zeros(shape=self.dims, dtype=np.float)).T


        if 'P' in kwargs.keys():
            self.P = kwargs['P']  # [Optional] Person ID
        else:
            self.P = np.matrix(np.eye(self.dims)) * 100

    def __update(self, meas):
        # UPDATE x, P based on measurement m
        # distance between measured and current position-belief

        y = np.matrix(meas).T - self.H * self.x
        S = self.H * self.P * self.H.T + self.R  # residual convariance
        K = self.P * self.H.T * S.I  # Kalman gain
        x_dash = self.x + K * y
        I = np.matrix(np.eye(self.F.shape[0]))  # identity matrix
        P_dash = (I - K * self.H) * self.P

        return x_dash, P_dash

    def predict_next(self, meas):
        x_dash, P_dash = self.__update(meas)

        # PREDICT x, P based on motion
        self.x = self.F*x_dash + self.motion
        self.P = self.F*P_dash*self.F.T + self.Q

        return np.asarray(self.x).T

    def fit(self, measurements):
        for meas in measurements:
            x_dash, P_dash = self.__update(meas)

            # PREDICT x, P based on motion
            self.x = self.F * x_dash + self.motion
            self.P = self.F * P_dash * self.F.T + self.Q

# a1 = Group(0)
#
#
# b1 = Person(personID=11, feat_vec=[1,2,3,4])
# # b2 = Person(12, [2,3,4,4])
#
# a1 + b1  # Add person to group
# b1 + Person(feat_vec=[2,3,4,4])  # Merge people
#
#
#
# a2 = Group(1)
# c1 = Person(personID=21, feat_vec=[11,12,13,14])
# c2 = Person(personID=22, feat_vec=[12,13,14,15])
# c3 = Person(personID=0, feat_vec=[22,123,124,125])
#
#
# print c1
# print c2
# a2 + c1  # Add person to group
# c1 + c2  # Merge people
#
# print c2
# a1 + c3

# print a1
# print a2

# print a1.members[0].personID
# print a1[11].fv
