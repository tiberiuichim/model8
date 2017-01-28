from sqlalchemy import func
from cornice import Service
from cornice.resource import resource
from pyramid.view import notfound_view_config
from .models import MLModel, DataFragment


@notfound_view_config(renderer='json')
def notfound_view(request):
    request.response.status = 404
    return {'status': 'notfound'}


def get_model_labels(model, session):
    res = (session.query(DataFragment.label, func.count(DataFragment.id))
           .select_from(MLModel)
           .join(MLModel.fragments)
           .filter(MLModel.name == model.name)
           .group_by(DataFragment.label)
           .all())
    return dict(res)      # a list like [('approved', 3), ('rejected', 6)]


@resource(collection_path="/",
          path="/{id}", cors_origins=('*',), cors_max_age=3600)
class MLModelResource(object):

    def __init__(self, request):
        self.request = request

    def _model_url(self, ml):
        return self.request.route_url(self.__class__.__name__.lower(),
                                      id=ml.name)

    def serialize_model(self, ml, sess):
        res = {}
        res['name'] = ml.name
        res['labels'] = get_model_labels(ml, sess)
        res['can_predict'] = ml.can_predict()
        res['url'] = self._model_url(ml)
        return res

    def collection_get(self):
        sess = self.request.dbsession
        res = []
        for ml in sess.query(MLModel):
            res.append(self.serialize_model(ml, sess))
        return {'models': res}

    def collection_post(self):
        data = self.request.json_body
        model = MLModel(name=data['name'])
        self.request.dbsession.add(model)
        return self.collection_get()

    def get(self):
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter(MLModel.name == name).one()
        return self.serialize_model(model, sess)

    def put(self):
        """ Add new data to the model. Expects :text and :label
        """
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter(MLModel.name == name).one()
        data = self.request.json_body
        text = data['text']
        label = data['label']
        frag = DataFragment(text=text, label=label)
        model.fragments.append(frag)
        return self.collection_get()

    def post(self):
        """ Compile the tokenizer, fit the model
        """
        sess = self.request.dbsession
        name = self.request.matchdict['id']
        model = sess.query(MLModel).filter_by(name=name).one()
        try:
            model.build()
            return {'acknowledge': True}
        except ValueError as e:
            return {'error': e.args[0]}


prophet = Service(name="prophetservice",
                  description="Use the prediction service",
                  path="/{name}/prophet",
                  cors_origins=('*',),
                  cors_max_age=3600)


@prophet.post()
def model_predict(request):
    # make predictions on one sentence
    sess = request.dbsession
    name = request.matchdict['name']
    text = request.json['text'].encode('utf-8')
    model = sess.query(MLModel).filter_by(name=name).one()
    if not model.can_predict():
        raise ValueError("Model is not able to predict")
    res = model.predict(text)
    return res


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
