import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import .utils as utils


def model_train(model, sess, data_train, KB_tuple):
    batched_data = utils.gen_batched_data(data_train, KB_tuple, FLAGS)
    outputs = model.step_train(sess, batched_data)
    sentence_ppx = np.sum(outputs[0])
    decoder_loss = np.sum(outputs[1])
    return sentence_ppx, decoder_loss


def generate_summary(model, sess, data_train, KB_tuple):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
    outputs = model.step_train(sess, batched_data, forward_only=True, summary=True)
    summary = outputs[-1]
    return summary


def model_evaluate(model, sess, data_dev, summary_writer, KB_tuple):
    ppx = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
        outputs = model.step_train(sess, batched_data, forward_only=True)
        ppx += np.sum(outputs[0])
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    ppx /= len(data_dev)
    summary = tf.Summary()
    summary.value.add(tag='perplexity/dev', simple_value=np.exp(ppx))
    summary_writer.add_summary(summary, model.global_step.eval())
    print("   perplexity on dev set: %.2f" % np.exp(ppx))


def get_steps(train_dir):
    steps, metafiles, datafiles, indexfiles = [], [], [], []
    for root, dirs, files in os.walk(train_dir):
        if root == train_dir:
            filenames = files
            for filename in filenames:
                if 'meta' in filename:
                    metafiles.append(filename)
                if 'data' in filename:
                    datafiles.append(filename)
                if 'index' in filename:
                    indexfiles.append(filename)
    metafiles.sort()
    datafiles.sort()
    indexfiles.sort(reverse=True)
    for f in indexfiles:
        steps.append(int(f[11:-6]))

    return steps


def model_test(sess, saver, data_dev, KB_tuple, setnum=5000, max_step=800000):
    with open('%s/stopwords' % FLAGS.data_dir) as f:
        stopwords = json.loads(f.readline())
    steps = get_steps(FLAGS.train_dir)
    results = []

    with open('%s/test.res' % FLAGS.train_dir, 'w') as resfile, open('%s/test.log' % FLAGS.train_dir, 'w') as outfile:
        steps = [step for step in steps if step in range(max_step)]
        for step in steps:
            outfile.write('test for model-%d\n' % step)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, step)
            print('restore from %s' % model_path)
            try:
                saver.restore(sess, model_path)
            except:
                continue
            st, ed = 0, FLAGS.batch_size
            loss = []
            while st < len(data_dev):
                selected_data = data_dev[st:ed]
                batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
                responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'],
                                               {'enc_inps:0': batched_data['posts'],
                                                'enc_lens:0': batched_data['posts_length'],
                                                'dec_inps:0': batched_data['responses'],
                                                'dec_lens:0': batched_data['responses_length'],
                                                'entities:0': batched_data['entities'],
                                                'triples:0': batched_data['triples'],
                                                'match_triples:0': batched_data['match_triples'],
                                                'enc_triples:0': batched_data['posts_triple'],
                                                'dec_triples:0': batched_data['responses_triple']})
                loss += [x for x in ppx_loss]
                for response in responses:
                    result = []
                    for token in response:
                        token = str(token, encoding='utf-8')
                        if token != '_EOS':
                            result.append(token)
                        else:
                            break
                    results.append(result)
                st, ed = ed, ed+FLAGS.batch_size
            match_entity_sum = [.0] * 4
            cnt = 0
            posts = [data['post'] for data in data_dev]
            responses = [data['response'] for data in data_dev]
            match_triples = [data['match_triples'] for data in data_dev]
            all_triples = [data['all_triples'] for data in data_dev]
            all_entites = [data['all_entities'] for data in data_dev]

            hypotheses = []
            references = []

            for post, response, result, match_triple, triples, entities in \
                    zip(posts, responses, results, match_triples, all_triples, all_entites):
                setidx = int(cnt / setnum)
                result_matched_entities = []
                entities = [KB_tuple["csk_entities"][x] for entity in entities for x in entity]
                for word in result:
                    if word not in stopwords and word in entities:
                        result_matched_entities.append(word)

                post = " ".join([str(p) for p in post])
                response = [str(r) for r in response]
                references.append([response])
                response = " ".join(response)
                
                hypotheses.append(result)
                result = " ".join(result)

                outfile.write('post: %s\nresponse: %s\nresult: %s\nmatch_entity: %s\n\n' %
                              (post, response, result, " ".join(result_matched_entities)))
                match_entity_sum[setidx] += len(set(result_matched_entities))
                cnt += 1
            match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
            losses = [np.sum(loss[x:x+setnum]) / float(setnum) for x in range(0, setnum*4, setnum)] + \
                     [np.sum(loss) / float(setnum*4)]
            losses = [np.exp(x) for x in losses]
            
            bleus_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
            bleus_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
            bleus_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
            bleus_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
            bleus_overall = corpus_bleu(references, hypotheses)
            bleus = [bleus_1, bleus_2, bleus_3, bleus_4, bleus_overall]

            def show(x):
                return ', '.join([str(v) for v in x])

            outfile.write('model: %d\n\tbleu: %s\n\tperplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' %
                          (step, bleus, show(losses), show(match_entity_sum), '=' * 50))
            resfile.write('model: %d\n\tbleu: %s\n\tperplexity: %s\n\tmatch_entity_rate: %s\n\n' %
                          (step, bleus, show(losses), show(match_entity_sum)))
            outfile.flush()
            resfile.flush()
