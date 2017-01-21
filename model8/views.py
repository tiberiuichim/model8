from cornice import Service
from cornice.resource import resource
from pyramid.view import notfound_view_config
from .models import MLModel, DataFragment


@notfound_view_config(renderer='json')
def notfound_view(request):
    request.response.status = 404
    return {'status': 'notfound'}


def get_model_labels(model, session):
    res = session.query(DataFragment.label).select_from(MLModel).\
        join(MLModel.fragments).filter(MLModel.name == model.name).\
        group_by(DataFragment.label).all()
    return [x[0] for x in res]


@resource(collection_path="/", path="/{id}")
class MLModelResource(object):
    def __init__(self, request):
        self.request = request

    def collection_get(self):
        sess = self.request.dbsession
        res = []
        for ml in sess.query(MLModel):
            d = {}
            d['name'] = ml.name
            d['labels'] = get_model_labels(ml, sess)
            res.append(d)
        return {'models': res}

    def collection_post(self):
        data = self.request.json_body
        model = MLModel(name=data['name'])
        self.request.dbsession.add(model)
        return {'name': model.name, 'labels': []}

    def get(self):
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter(MLModel.name == name).one()
        labels = get_model_labels(model, sess)
        res = {'name': name, 'labels': labels}
        return res

    def put(self):
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter(MLModel.name == name).one()
        data = self.request.json_body
        text = data['text']
        label = data['label']
        frag = DataFragment(text=text, label=label)
        model.fragments.append(frag)
        return self.get()

    def post(self):
        """ Compile the tokenizer, fit the model
        """
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter_by(name=name).one()
        model.build()
        return {'acknowledge': True}


prophet = Service(name="prophetservice",
                  description="Use the prediction service",
                  path="/{name}/prophet")


@prophet.post()
def model_predict(request):
    # make predictions on each line
    return {}


@prophet.get()
def get_status(request):
    return {}


# tokenizerservice = Service(name="tokenizerservice",
#                            description="Interact with a tokenizer",
#                            path="/{name}/_tokenizer"
#                            )
#
#
# @tokenizerservice.delete()
# def tokenizer_delete(request):
#     return {'acknowledge': True}
#
#
# @tokenizerservice.post()
# def tokenizer_compile(request):
#     # set status = 'building'
#     return {'acknowledge': True}
#
#
# @tokenizerservice.get()
# def tokenizer_download(request):
#     # return the model hp5
#     return {'acknowledge': True}
#
#
# modelservice = Service(name="modelservice",
#                        description="Interact with a model",
#                        path="/{name}/_model"
#                        )
#
#
# @modelservice.post()
# def model_compile(request):
#     return {'acknowledge': True}
#
#
# @modelservice.delete()
# def model_delete(request):
#     return {'acknowledge': True}
#
#
# @modelservice.get()
# def model_download(request):
#     # return the model hp5
#     return {'acknowledge': True}
#
#
