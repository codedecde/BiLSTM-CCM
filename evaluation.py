import sys,pdb,copy,munkres
import argparse
'''
    python evaluation.py  -p_file <Prediction Label File> -g_file <Gold Label File> -s_file <Output Label File> (optional)
'''


class post(object):
    def __init__(self):
        self.label2idx = {'type': 0, 'attr': 1, 'location': 2, 'temporal': 3}
        self.raw_post = []
        self.post_with_features = []
        self.gold_instances = None
        self.prediction_instances = None
        self.precision_list = [None for idx in range(len(self.label2idx))]
        self.recall_list = [None for idx in range(len(self.label2idx))]
        # TODO : Remove this
        self.precision_list_mm = [None for idx in range(len(self.label2idx))]
        self.recall_list_mm = [None for idx in range(len(self.label2idx))]
        self.mode = 'multi_match'  # set to max_match for maximal_match
        self.temporal = False

    def longest_common_subsequence(self, post1, post2):
        post1 = post1.split(' ')
        post2 = post2.split(' ')
        n = len(post1)
        m = len(post2)
        lcs = [[0 for idx in range(m + 1)] for idx in range(n + 1)]

        for idx in range(1, n + 1):
            for jdx in range(1, m + 1):
                if post1[idx - 1] == post2[jdx - 1]:
                    lcs[idx][jdx] = 1 + lcs[idx - 1][jdx - 1]
                else:
                    lcs[idx][jdx] = max(lcs[idx - 1][jdx], lcs[idx][jdx - 1])

        return lcs[n][m]

    def removeDuplicates(self, instance_list):
        new_instance_list = []
        # pdb.set_trace()
        add_to_list = [True for idx in range(len(instance_list))]
        for idx in range(len(instance_list) - 1):
            for jdx in range(idx + 1, len(instance_list)):
                lcs = self.longest_common_subsequence(
                    instance_list[idx], instance_list[jdx])
                if lcs == len(instance_list[idx].split(' ')):
                    add_to_list[idx] = False
                elif lcs == len(instance_list[jdx].split(' ')):
                    add_to_list[jdx] = False
        for idx in range(len(instance_list)):
            if add_to_list[idx]:
                new_instance_list.append(instance_list[idx])
        return new_instance_list

    def set_values(self, post, prediction):
        post = post.split('\n')
        self.post_with_features = copy.deepcopy(post)
        prediction = prediction.split('\n')
        assert(len(post) == len(prediction))
        self.gold_instances = [[] for idx in range(len(self.label2idx))]
        self.prediction_instances = [[] for idx in range(len(self.label2idx))]
        curr_gold_label, curr_gold_instance, curr_predicted_label, curr_predicted_instance = "", "", "", ""

        for idx in range(len(post)):
            word = post[idx].split(' ')[0]
            gold_label = post[idx].split(' ')[-1]
            predicted_label = prediction[idx]
            self.raw_post.append((word, predicted_label, gold_label))
            if gold_label != curr_gold_label:
                if curr_gold_instance != "":
                    if curr_gold_label in self.label2idx:
                        self.gold_instances[self.label2idx[curr_gold_label]].append(curr_gold_instance)
                curr_gold_instance = word
                curr_gold_label = gold_label
            else:
                curr_gold_instance += ' ' + word
            if predicted_label != curr_predicted_label:
                if curr_predicted_instance != "":
                    if curr_predicted_label in self.label2idx:
                        self.prediction_instances[self.label2idx[curr_predicted_label]].append(curr_predicted_instance)
                curr_predicted_instance = word
                curr_predicted_label = predicted_label
            else:
                curr_predicted_instance += ' ' + word
        if curr_predicted_instance != '' and curr_predicted_label in self.label2idx:
            self.prediction_instances[self.label2idx[curr_predicted_label]].append(curr_predicted_instance)
        if curr_gold_instance != '' and curr_gold_label in self.label2idx:
            self.gold_instances[self.label2idx[curr_gold_label]].append(curr_gold_instance)

        # Removal by max sequence match
        # self.prediction_instances = [self.removeDuplicates(elem) for elem in self.prediction_instances]
        # self.gold_instances = [self.removeDuplicates(elem) for elem in self.gold_instances]
        self.prediction_instances = [list(set(elem)) for elem in self.prediction_instances]
        self.gold_instances = [list(set(elem)) for elem in self.gold_instances]
        # Now computing the fractional counts for precision and recall
        for idx in range(len(self.prediction_instances)):
            if self.mode == 'multi_match':
                # Multi match
                self.precision_list[idx] = self.multi_match(
                    self.prediction_instances[idx], self.gold_instances[idx], True)
                self.recall_list[idx] = self.multi_match(
                    self.prediction_instances[idx], self.gold_instances[idx], False)
                self.precision_list_mm[idx] = self.maximal_match(
                    self.prediction_instances[idx], self.gold_instances[idx], True)
                self.recall_list_mm[idx] = self.maximal_match(
                    self.prediction_instances[idx], self.gold_instances[idx], False)
            elif self.mode == 'max_match':
                # Maximal match
                self.precision_list[idx] = self.maximal_match(
                    self.prediction_instances[idx], self.gold_instances[idx], True)
                self.recall_list[idx] = self.maximal_match(
                    self.prediction_instances[idx], self.gold_instances[idx], False)
            else:
                raise ValueError

    def multi_match(self, pred_list, gold_list, precision=True):
        # pdb.set_trace()
        n = len(pred_list)
        m = len(gold_list)
        # Base Cases:
        # 1.1 no gold:
        if m == 0:
            if precision:
                return [(0., pred_list[idx], '') for idx in range(n)]
            else:
                return []
        # 1.2 no pred:
        if n == 0:
            if precision:
                return []
            else:
                return [(0., '', gold_list[idx]) for idx in range(m)]
        if precision:
            ret = []
            for idx in range(n):
                max_match = -1
                match_idx = -1
                for jdx in range(m):
                    lcs = self.longest_common_subsequence(pred_list[idx], gold_list[jdx])
                    if max_match < lcs:
                        max_match = lcs
                        match_idx = jdx
                ret.append((float(max_match) / len(pred_list[idx].split(' ')), pred_list[idx], gold_list[match_idx]))
            return ret
        else:
            ret = []
            for idx in range(m):
                max_match = 0
                match_idx = -1
                for jdx in range(n):
                    lcs = self.longest_common_subsequence(gold_list[idx], pred_list[jdx])
                    if max_match < lcs:
                        max_match = lcs
                        match_idx = jdx
                ret.append((float(max_match) / len(gold_list[idx].split(' ')), pred_list[match_idx], gold_list[idx]))
            return ret

    def maximal_match(self, pred_list, gold_list, precision=True):
        n = len(pred_list)
        m = len(gold_list)
        # Base Cases:
        # 1.1 no gold:
        if m == 0:
            if precision:
                return [(0., pred_list[idx], '') for idx in range(n)]
            else:
                return []
        # 1.2 no pred:
        if n == 0:
            if precision:
                return []
            else:
                return [(0., '', gold_list[idx]) for idx in range(m)]

        edges = [[0. for idx in range(m)] for jdx in range(n)]
        for idx in range(n):
            for jdx in range(m):
                edges[idx][jdx] = float(self.longest_common_subsequence(pred_list[idx], gold_list[jdx]))
                if precision:
                    edges[idx][jdx] /= len(pred_list[idx].split(' '))
                else:
                    edges[idx][jdx] /= len(gold_list[jdx].split(' '))
        return self.maximal_match_helper(edges, pred_list, gold_list, precision)

    def maximal_match_helper(self, edges, pred_list, gold_list, row_bool):
        act_row = len(edges)
        act_col = len(edges[0])
        n = max(act_row, act_col)
        large_val = max([max(elem) for elem in edges])
        large_val = (large_val + 1) * (large_val + 1)
        m = munkres.Munkres()
        cost = [[large_val for idx in range(n)] for idx in range(n)]
        for idx in range(act_row):
            for jdx in range(act_col):
                cost[idx][jdx] = large_val - edges[idx][jdx]
        indexes = m.compute(cost)
        ret = []
        for row, col in indexes:
            if row < act_row and col < act_col:
                ret.append((large_val - cost[row][col], pred_list[row], gold_list[col]))
            else:
                if row < act_row:
                    ret.append((large_val - cost[row][col], pred_list[row], ''))
                elif col < act_col:
                    ret.append((large_val - cost[row][col], '', gold_list[col]))

        ret = sorted(ret, reverse=True)
        if row_bool:
            ret = ret[:act_row]
        else:
            ret = ret[:act_col]
        return ret

    def __str__(self, idx=None):
        write_buf = '--------------------------- POST ----------------------\n'
        if idx is not None:
            write_buf += 'IDX: ' + str(idx) + '\n'
        write_buf += ' '.join(map(lambda x: x[0], self.raw_post)) + '\n'
        write_buf += 'TYPE: \n'
        write_buf += '\t GOLD: %s\n' % str(self.gold_instances[0])
        write_buf += '\t PRED: %s\n' % str(self.prediction_instances[0])
        write_buf += '\t PRECISION COUNTS: %s\n' % str(self.precision_list[0])
        write_buf += '\t RECALL COUNTS: %s\n' % str(self.recall_list[0])

        write_buf += 'ATTR: \n'
        write_buf += '\t GOLD: %s\n' % str(self.gold_instances[1])
        write_buf += '\t PRED: %s\n' % str(self.prediction_instances[1])
        write_buf += '\t PRECISION COUNTS: %s\n' % str(self.precision_list[1])
        write_buf += '\t RECALL COUNTS: %s\n' % str(self.recall_list[1])

        write_buf += 'LOCATION: \n'
        write_buf += '\t GOLD: %s\n' % str(self.gold_instances[2])
        write_buf += '\t PRED: %s\n' % str(self.prediction_instances[2])
        write_buf += '\t PRECISION COUNTS: %s\n' % str(self.precision_list[2])
        write_buf += '\t RECALL COUNTS: %s\n' % str(self.recall_list[2])

        if self.temporal:
            write_buf += 'TEMPORAL: \n'
            write_buf += '\t GOLD: %s\n' % str(self.gold_instances[3])
            write_buf += '\t PRED: %s\n' % str(self.prediction_instances[3])
            write_buf += '\t PRECISION COUNTS: %s\n' % str(self.precision_list[3])
            write_buf += '\t RECALL COUNTS: %s\n' % str(self.recall_list[3])
        write_buf += '-------------------------------------------------------\n'
        return write_buf


class evaluator(object):
    def __init__(self, pred_file_name, gold_file_name, verbosity=True):
        posts = filter(lambda x: x != '', open(gold_file_name).read().split('\n\n'))
        predictions = filter(lambda x: x != '', open(pred_file_name).read().split('\n\n\n'))
        # pdb.set_trace()
        assert(len(posts) == len(predictions))
        self.posts = [None for idx in range(len(posts))]
        idx = 0
        for _post, _prediction in zip(posts, predictions):
            self.posts[idx] = post()
            self.posts[idx].set_values(_post, _prediction)
            idx += 1
        self.verbose = verbosity
        self.temporal = False

    def compute_averages(self):
        label2idx = {'type': 0, 'attr': 1, 'location': 2, 'temporal': 3}
        self._precision_list = [0. for idx in range(len(label2idx))]
        self._recall_list = [0. for idx in range(len(label2idx))]
        self.average_per_post_precision_list = [0. for idx in range(len(label2idx))]
        self.average_per_post_recall_list = [0. for idx in range(len(label2idx))]
        for idx in range(len(label2idx)):
            macro_count_prec = 0
            macro_count_recall = 0
            micro_count_prec = 0
            micro_count_recall = 0
            for post in self.posts:
                x = sum([x[0] for x in post.precision_list[idx]])
                xl = len(post.precision_list[idx])
                y = sum([x[0] for x in post.recall_list[idx]])
                yl = len(post.recall_list[idx])
                self._precision_list[idx] += x
                self._recall_list[idx] += y
                if xl != 0:
                    self.average_per_post_precision_list[idx] += (float(x) / xl)
                else:
                    assert(x == 0)
                if yl != 0:
                    self.average_per_post_recall_list[idx] += (float(y) / yl)
                else:
                    assert(y == 0)
                macro_count_prec += xl
                macro_count_recall += yl
                micro_count_prec += min(1, xl)
                micro_count_recall += min(1, yl)
            if macro_count_prec != 0:
                self._precision_list[idx] /= macro_count_prec
            if macro_count_recall != 0:
                self._recall_list[idx] /= macro_count_recall
            if micro_count_prec != 0:
                self.average_per_post_precision_list[idx] /= micro_count_prec
            if micro_count_recall != 0:
                self.average_per_post_recall_list[idx] /= micro_count_recall

    def __str__(self):
        write_buf = ''
        if self.verbose:
            for post in self.posts:
                write_buf += str(post)

        write_buf += '---------------------- UNWEIGHTED  --------------------\n'
        write_buf += 'TYPE: \n'
        write_buf += '\t   PREC: %.4f\n' % self._precision_list[0]
        write_buf += '\t    REC: %.4f\n' % self._recall_list[0]
        write_buf += '\t FSCORE: %.4f\n' % (
            2 * self._precision_list[0] * self._recall_list[0] / (self._precision_list[0] + self._recall_list[0]))

        write_buf += 'ATTR: \n'
        write_buf += '\t   PREC: %.4f\n' % self._precision_list[1]
        write_buf += '\t    REC: %.4f\n' % self._recall_list[1]
        write_buf += '\t FSCORE: %.4f\n' % (
            2 * self._precision_list[1] * self._recall_list[1] / (self._precision_list[1] + self._recall_list[1]))

        write_buf += 'LOCATION: \n'
        write_buf += '\t   PREC: %.4f\n' % self._precision_list[2]
        write_buf += '\t    REC: %.4f\n' % self._recall_list[2]
        write_buf += '\t FSCORE: %.4f\n' % (
            2 * self._precision_list[2] * self._recall_list[2] / (self._precision_list[2] + self._recall_list[2]))

        if self.temporal:
            write_buf += 'TEMPORAL: \n'
            write_buf += '\t   PREC: %.4f\n' % self._precision_list[3]
            write_buf += '\t    REC: %.4f\n' % self._recall_list[3]
            if (self._precision_list[3] + self._recall_list[3]) != 0.:
                write_buf += '\t FSCORE: %.4f\n' % (
                    (2 * self._precision_list[3] * self._recall_list[3]) /
                    (self._precision_list[3] + self._recall_list[3])
                )
            else:
                write_buf += '\t FSCORE: NAN\n'

        write_buf += '----------------------  WEIGHTED  ---------------------\n'

        write_buf += 'TYPE: \n'
        write_buf += '\t   PREC: %.4f\n' % self.average_per_post_precision_list[0]
        write_buf += '\t    REC: %.4f\n' % self.average_per_post_recall_list[0]
        write_buf += '\t FSCORE: %.4f\n' % (
            (2 * self.average_per_post_precision_list[0] * self.average_per_post_recall_list[0]) /
            (self.average_per_post_precision_list[0] + self.average_per_post_recall_list[0]))

        write_buf += 'ATTR: \n'
        write_buf += '\t   PREC: %.4f\n' % self.average_per_post_precision_list[1]
        write_buf += '\t    REC: %.4f\n' % self.average_per_post_recall_list[1]
        write_buf += '\t FSCORE: %.4f\n' % (
            (2 * self.average_per_post_precision_list[1] * self.average_per_post_recall_list[1]) /
            (self.average_per_post_precision_list[1] + self.average_per_post_recall_list[1]))

        write_buf += 'LOCATION: \n'
        write_buf += '\t   PREC: %.4f\n' % self.average_per_post_precision_list[2]
        write_buf += '\t    REC: %.4f\n' % self.average_per_post_recall_list[2]
        write_buf += '\t FSCORE: %.4f\n' % (
            (2 * self.average_per_post_precision_list[2] * self.average_per_post_recall_list[2]) /
            (self.average_per_post_precision_list[2] + self.average_per_post_recall_list[2]))
        if self.temporal:
            write_buf += 'TEMPORAL: \n'
            write_buf += '\t   PREC: %.4f\n' % self.average_per_post_precision_list[3]
            write_buf += '\t    REC: %.4f\n' % self.average_per_post_recall_list[3]
            if self.average_per_post_precision_list[3] + self.average_per_post_recall_list[3] != 0.:
                write_buf += '\t FSCORE: %.4f\n' % (
                    (2 * self.average_per_post_precision_list[3] * self.average_per_post_recall_list[3]) /
                    (self.average_per_post_precision_list[3] + self.average_per_post_recall_list[3]))
            else:
                write_buf += '\t FSCORE: NAN\n'
        write_buf += '-------------------------------------------------------\n'
        return write_buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluator Function')
    parser.add_argument('-p_file', dest='p_file', type=str,
                        default='', help='The Prediction File')
    parser.add_argument('-g_file', dest='g_file', type=str,
                        default='', help='The Gold File')
    parser.add_argument('-s_file', dest='save_file', type=str,
                        default='', help='File to save in')
    parser.add_argument('-verbose', dest='verbose', type=str,
                        default='True', help='Info to print')
    args = parser.parse_args()
    assert(args.p_file != '' and args.g_file != '')
    if args.verbose == 'False':
        evl = evaluator(args.p_file, args.g_file, False)
    else:
        evl = evaluator(args.p_file, args.g_file)
    evl.compute_averages()
    # label2idx = {'type':0, 'attr':1, 'location':2, 'temporal':3}
    # for post in evl.posts:
    #   # Precision Difference
    #   for old_method, new_method in zip(sorted(post.recall_list_mm[1]), sorted(post.recall_list[1])):
    #       if abs(old_method[0] - new_method[0]) > 0.00001:
    #           print ' '.join([ elem[0] for elem in post.raw_post])
    #           print '\n\n'
    if args.save_file == '':
        print(evl)
    else:
        open(args.save_file, 'wb').write(str(evl))
